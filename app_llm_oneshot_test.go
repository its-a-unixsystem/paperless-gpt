package main

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"testing"
	"text/template"

	"github.com/Masterminds/sprig/v3"
	"github.com/sirupsen/logrus"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
	"google.golang.org/genai"
)

func TestOneshotResponseParsing(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		wantErr  bool
		validate func(t *testing.T, resp oneshotResponse)
	}{
		{
			name: "valid full response",
			input: `{
				"title": "Invoice 2024-001",
				"tags": ["invoice", "finance"],
				"correspondent": "Acme Corp",
				"document_type": "Invoice",
				"created_date": "2024-01-15",
				"custom_fields": [{"field": "Invoice Number", "value": "INV-2024-001"}]
			}`,
			wantErr: false,
			validate: func(t *testing.T, resp oneshotResponse) {
				assert.Equal(t, "Invoice 2024-001", resp.Title)
				assert.Equal(t, []string{"invoice", "finance"}, resp.Tags)
				assert.Equal(t, "Acme Corp", resp.Correspondent)
				assert.Equal(t, "Invoice", resp.DocumentType)
				assert.Equal(t, "2024-01-15", resp.CreatedDate)
				require.Len(t, resp.CustomFields, 1)
				assert.Equal(t, "Invoice Number", resp.CustomFields[0].Field)
				assert.Equal(t, "INV-2024-001", resp.CustomFields[0].Value)
			},
		},
		{
			name: "partial response - only title and tags",
			input: `{
				"title": "My Document",
				"tags": ["general"]
			}`,
			wantErr: false,
			validate: func(t *testing.T, resp oneshotResponse) {
				assert.Equal(t, "My Document", resp.Title)
				assert.Equal(t, []string{"general"}, resp.Tags)
				assert.Empty(t, resp.Correspondent)
				assert.Empty(t, resp.DocumentType)
				assert.Empty(t, resp.CreatedDate)
				assert.Nil(t, resp.CustomFields)
			},
		},
		{
			name: "empty tags array",
			input: `{
				"title": "Test",
				"tags": []
			}`,
			wantErr: false,
			validate: func(t *testing.T, resp oneshotResponse) {
				assert.Equal(t, "Test", resp.Title)
				assert.Empty(t, resp.Tags)
			},
		},
		{
			name:    "malformed JSON",
			input:   `{not valid json`,
			wantErr: true,
		},
		{
			name:    "empty string",
			input:   ``,
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var resp oneshotResponse
			err := json.Unmarshal([]byte(tt.input), &resp)
			if tt.wantErr {
				assert.Error(t, err)
			} else {
				require.NoError(t, err)
				tt.validate(t, resp)
			}
		})
	}
}

func TestExtractOneshotTextFromResponse(t *testing.T) {
	logger := logrus.New()
	logger.SetOutput(io.Discard)
	docLogger := logrus.NewEntry(logger)

	t.Run("nil response", func(t *testing.T) {
		text, err := extractOneshotTextFromResponse(nil, "model", 123, docLogger)
		assert.Empty(t, text)
		assert.Error(t, err)
		assert.False(t, isPermanentOneshotError(err))
	})

	t.Run("non-text first part returns error", func(t *testing.T) {
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					FinishReason: genai.FinishReasonSafety,
					Content: &genai.Content{
						Parts: []*genai.Part{
							{FunctionCall: &genai.FunctionCall{}},
						},
					},
				},
			},
		}

		text, err := extractOneshotTextFromResponse(resp, "model", 123, docLogger)
		assert.Empty(t, text)
		assert.Error(t, err)
		assert.True(t, isPermanentOneshotError(err))
	})

	t.Run("empty text returns retryable error", func(t *testing.T) {
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					FinishReason: genai.FinishReasonStop,
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "   "},
						},
					},
				},
			},
		}

		text, err := extractOneshotTextFromResponse(resp, "model", 123, docLogger)
		assert.Empty(t, text)
		assert.Error(t, err)
		assert.False(t, isPermanentOneshotError(err))
	})

	t.Run("text part returns text", func(t *testing.T) {
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					FinishReason: genai.FinishReasonStop,
					Content: &genai.Content{
						Parts: []*genai.Part{
							{Text: "{\"ok\":true}"},
						},
					},
				},
			},
		}

		text, err := extractOneshotTextFromResponse(resp, "model", 123, docLogger)
		assert.NoError(t, err)
		assert.Equal(t, "{\"ok\":true}", text)
	})

	t.Run("uses later text part when first part is non-text", func(t *testing.T) {
		resp := &genai.GenerateContentResponse{
			Candidates: []*genai.Candidate{
				{
					FinishReason: genai.FinishReasonStop,
					Content: &genai.Content{
						Parts: []*genai.Part{
							{FunctionCall: &genai.FunctionCall{}},
							{Text: "{\"ok\":true}"},
						},
					},
				},
			},
		}

		text, err := extractOneshotTextFromResponse(resp, "model", 123, docLogger)
		assert.NoError(t, err)
		assert.Equal(t, "{\"ok\":true}", text)
	})
}

func TestFilterTagsAgainstAvailable(t *testing.T) {
	available := []string{"Invoice", "Finance", "Medical", "Tax", "Insurance"}

	tests := []struct {
		name      string
		suggested []string
		want      []string
	}{
		{
			name:      "all tags match",
			suggested: []string{"Invoice", "Finance"},
			want:      []string{"Invoice", "Finance"},
		},
		{
			name:      "some tags match",
			suggested: []string{"Invoice", "NonExistent", "Tax"},
			want:      []string{"Invoice", "Tax"},
		},
		{
			name:      "case insensitive match",
			suggested: []string{"invoice", "FINANCE"},
			want:      []string{"Invoice", "Finance"},
		},
		{
			name:      "no tags match",
			suggested: []string{"Unknown", "Other"},
			want:      []string{},
		},
		{
			name:      "empty suggested",
			suggested: []string{},
			want:      []string{},
		},
		{
			name:      "nil suggested",
			suggested: nil,
			want:      []string{},
		},
		{
			name:      "tags with whitespace",
			suggested: []string{" Invoice ", "  Finance"},
			want:      []string{"Invoice", "Finance"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := filterTagsAgainstAvailable(tt.suggested, available)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestValidateDocumentType(t *testing.T) {
	available := []string{"Invoice", "Letter", "Receipt", "Contract"}

	tests := []struct {
		name      string
		suggested string
		want      string
	}{
		{
			name:      "exact match",
			suggested: "Invoice",
			want:      "Invoice",
		},
		{
			name:      "case insensitive match",
			suggested: "invoice",
			want:      "Invoice",
		},
		{
			name:      "not in list",
			suggested: "Unknown Type",
			want:      "",
		},
		{
			name:      "empty string",
			suggested: "",
			want:      "",
		},
		{
			name:      "whitespace trimmed",
			suggested: "  Invoice  ",
			want:      "Invoice",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := validateDocumentType(tt.suggested, available)
			assert.Equal(t, tt.want, got)
		})
	}
}

func TestValidateCustomFields(t *testing.T) {
	available := []CustomField{
		{ID: 1, Name: "Invoice Number", DataType: "string"},
		{ID: 2, Name: "Due Date", DataType: "date"},
		{ID: 3, Name: "Amount", DataType: "monetary"},
	}

	suggested := []struct {
		Field string      `json:"field"`
		Value interface{} `json:"value"`
	}{
		{Field: "Invoice Number", Value: "INV-001"},
		{Field: "Unknown Field", Value: "should be skipped"},
		{Field: "Amount", Value: "EUR100.50"},
	}

	result := validateCustomFields(suggested, available)

	require.Len(t, result, 2)
	assert.Equal(t, 1, result[0].ID)
	assert.Equal(t, "Invoice Number", result[0].Name)
	assert.Equal(t, "INV-001", result[0].Value)
	assert.Equal(t, 3, result[1].ID)
	assert.Equal(t, "Amount", result[1].Name)
	assert.Equal(t, "EUR100.50", result[1].Value)
}

func TestOneshotTemplateRendering(t *testing.T) {
	tmplContent := `You are a document processing assistant. You will be given a PDF document (attached as binary data).
Your task is to read and understand the document, then extract structured metadata from it.

The document content is likely in {{ .Language }}.
Today's date is {{ .Today }}.

**Original document title:** {{ .OriginalTitle }}
**Original document tags:** {{ .OriginalTags | join ", " }}

Please extract the following fields from the document and return them as a JSON object.
Only include fields that are requested below.

{{ if .GenerateTitle -}}
**Title:** Suggest a concise, descriptive title for this document. If the original title is already meaningful (not just a filename), you may use it as a reference.
{{ end -}}

{{ if .GenerateTags -}}
**Tags:** Select the most relevant tags from this list. Be selective — only choose tags that clearly apply. Return as a JSON array of strings.
<available_tags>
{{ .AvailableTags | join ", " }}
</available_tags>
{{ end -}}

{{ if .GenerateCorrespondent -}}
**Correspondent:** Identify the sender or main entity associated with this document. Prefer a match from the example list.
<example_correspondents>
{{ .AvailableCorrespondents | join ", " }}
</example_correspondents>
{{ if .CorrespondentBlackList -}}
<blacklisted_correspondents>
{{ .CorrespondentBlackList | join ", " }}
</blacklisted_correspondents>
{{ end -}}
{{ end -}}

{{ if .GenerateDocumentType -}}
**Document Type:** Select the most appropriate document type from this list.
<available_document_types>
{{ .AvailableDocumentTypes | join ", " }}
</available_document_types>
{{ end -}}

{{ if .GenerateCreatedDate -}}
**Created Date:** Determine the date the document was originally created or issued. Return in YYYY-MM-DD format.
{{ end -}}

{{ if .GenerateCustomFields -}}
**Custom Fields:** Extract values for the following custom fields.
{{ .CustomFieldsXML }}
{{ end -}}
`
	tmpl, err := template.New("oneshot").Funcs(sprig.FuncMap()).Parse(tmplContent)
	require.NoError(t, err)

	tests := []struct {
		name     string
		data     map[string]interface{}
		contains []string
		excludes []string
	}{
		{
			name: "all features enabled",
			data: map[string]interface{}{
				"Language":                "English",
				"Today":                   "2024-01-15",
				"OriginalTitle":           "test.pdf",
				"OriginalTags":            []string{"tag1", "tag2"},
				"GenerateTitle":           true,
				"GenerateTags":            true,
				"GenerateCorrespondent":   true,
				"GenerateDocumentType":    true,
				"GenerateCreatedDate":     true,
				"GenerateCustomFields":    true,
				"AvailableTags":           []string{"Invoice", "Finance"},
				"AvailableCorrespondents": []string{"Acme", "Globex"},
				"CorrespondentBlackList":  []string{"BadCo"},
				"AvailableDocumentTypes":  []string{"Invoice", "Letter"},
				"CustomFieldsXML":         `<custom_fields><field name="Amount" type="monetary"></field></custom_fields>`,
			},
			contains: []string{
				"**Title:**",
				"**Tags:**",
				"**Correspondent:**",
				"**Document Type:**",
				"**Created Date:**",
				"**Custom Fields:**",
				"Acme, Globex",
				"BadCo",
				"Invoice, Finance",
				"Invoice, Letter",
			},
		},
		{
			name: "only title enabled",
			data: map[string]interface{}{
				"Language":               "German",
				"Today":                  "2024-06-01",
				"OriginalTitle":          "doc.pdf",
				"OriginalTags":           []string{},
				"GenerateTitle":          true,
				"GenerateTags":           false,
				"GenerateCorrespondent":  false,
				"GenerateDocumentType":   false,
				"GenerateCreatedDate":    false,
				"GenerateCustomFields":   false,
				"AvailableTags":          []string{},
				"AvailableDocumentTypes": []string{},
				"CustomFieldsXML":        "",
			},
			contains: []string{"**Title:**", "German"},
			excludes: []string{"**Tags:**", "**Correspondent:**", "**Document Type:**", "**Created Date:**", "**Custom Fields:**"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var buf bytes.Buffer
			err := tmpl.Execute(&buf, tt.data)
			require.NoError(t, err)

			result := buf.String()
			for _, s := range tt.contains {
				assert.Contains(t, result, s)
			}
			for _, s := range tt.excludes {
				assert.NotContains(t, result, s)
			}
		})
	}
}

func TestStripMarkdownAndReasoning(t *testing.T) {
	tests := []struct {
		name  string
		input string
		want  string
	}{
		{
			name:  "json wrapped in markdown",
			input: "```json\n{\"title\": \"test\"}\n```",
			want:  `{"title": "test"}`,
		},
		{
			name:  "json with reasoning",
			input: "<think>I should extract the title</think>{\"title\": \"test\"}",
			want:  `{"title": "test"}`,
		},
		{
			name:  "plain json",
			input: `{"title": "test"}`,
			want:  `{"title": "test"}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stripReasoning(tt.input)
			result = stripMarkdown(result)
			assert.Equal(t, tt.want, result)
		})
	}
}

func TestIsPermanentOneshotErrorWrapped(t *testing.T) {
	baseErr := &permanentOneshotError{msg: "permanent failure"}
	wrappedErr := fmt.Errorf("wrapped: %w", baseErr)
	assert.True(t, isPermanentOneshotError(wrappedErr))
}
