package main

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"html"
	"os"
	"slices"
	"strings"
	"sync"

	"github.com/gabriel-vasile/mimetype"
	"github.com/sirupsen/logrus"
	"google.golang.org/genai"
)

// oneshotResponse is the JSON structure expected from the oneshot LLM response
type oneshotResponse struct {
	Content       string   `json:"content,omitempty"`
	Title         string   `json:"title,omitempty"`
	Tags          []string `json:"tags,omitempty"`
	Correspondent string   `json:"correspondent,omitempty"`
	DocumentType  string   `json:"document_type,omitempty"`
	CreatedDate   string   `json:"created_date,omitempty"`
	CustomFields  []struct {
		Field string      `json:"field"`
		Value interface{} `json:"value"`
	} `json:"custom_fields,omitempty"`
}

var (
	oneshotLLM     *genai.Client
	oneshotLLMOnce sync.Once
	oneshotLLMErr  error
)

// getOrCreateOneshotLLM lazily initializes the Google AI client for oneshot mode
func getOrCreateOneshotLLM() (*genai.Client, error) {
	oneshotLLMOnce.Do(func() {
		apiKey := os.Getenv("GOOGLEAI_API_KEY")
		if apiKey == "" {
			oneshotLLMErr = fmt.Errorf("GOOGLEAI_API_KEY is not set")
			return
		}

		client, err := genai.NewClient(context.Background(), &genai.ClientConfig{
			APIKey:  apiKey,
			Backend: genai.BackendGeminiAPI,
		})
		if err != nil {
			oneshotLLMErr = fmt.Errorf("failed to create oneshot Google AI client: %w", err)
			return
		}

		oneshotLLM = client
	})

	return oneshotLLM, oneshotLLMErr
}

// permanentOneshotError represents a non-retryable LLM error (e.g. content filtering)
type permanentOneshotError struct {
	msg string
}

func (e *permanentOneshotError) Error() string {
	return e.msg
}

func isPermanentOneshotError(err error) bool {
	var pe *permanentOneshotError
	return errors.As(err, &pe)
}

// permanentFinishReasons are finish reasons that will never succeed on retry with the same model
var permanentFinishReasons = map[genai.FinishReason]bool{
	genai.FinishReasonRecitation:        true,
	genai.FinishReasonSafety:            true,
	genai.FinishReasonBlocklist:         true,
	genai.FinishReasonProhibitedContent: true,
	genai.FinishReasonSPII:              true,
}

func oneshotPartKind(part *genai.Part) string {
	if part == nil {
		return "nil"
	}
	if strings.TrimSpace(part.Text) != "" {
		return "text"
	}
	if part.FunctionCall != nil {
		return "functionCall"
	}
	if part.FunctionResponse != nil {
		return "functionResponse"
	}
	if part.InlineData != nil {
		return "inlineData"
	}
	if part.FileData != nil {
		return "fileData"
	}
	if part.CodeExecutionResult != nil {
		return "codeExecutionResult"
	}
	if part.ExecutableCode != nil {
		return "executableCode"
	}
	if part.VideoMetadata != nil {
		return "videoMetadata"
	}
	if part.Thought {
		return "thought"
	}
	return "unknown"
}

func findOneshotTextPart(parts []*genai.Part) (string, *genai.Part) {
	var fallbackTextPart *genai.Part
	for _, part := range parts {
		if part == nil || strings.TrimSpace(part.Text) == "" {
			continue
		}
		if !part.Thought {
			return part.Text, part
		}
		if fallbackTextPart == nil {
			fallbackTextPart = part
		}
	}
	if fallbackTextPart != nil {
		return fallbackTextPart.Text, fallbackTextPart
	}
	return "", nil
}

func extractOneshotTextFromResponse(resp *genai.GenerateContentResponse, model string, inputSizeBytes int, docLogger *logrus.Entry) (string, error) {
	var candidate *genai.Candidate
	var firstPart *genai.Part
	var textPart *genai.Part
	var text string

	if resp != nil && len(resp.Candidates) > 0 {
		candidate = resp.Candidates[0]
		if candidate != nil && candidate.Content != nil && len(candidate.Content.Parts) > 0 {
			firstPart = candidate.Content.Parts[0]
			var selectedPart *genai.Part
			text, selectedPart = findOneshotTextPart(candidate.Content.Parts)
			if selectedPart != nil {
				textPart = selectedPart
			}
		}
	}

	if textPart != nil {
		return text, nil
	}

	// Log diagnostic details
	var finishReason genai.FinishReason
	if resp != nil {
		if resp.PromptFeedback != nil {
			docLogger.Errorf("[%s] Prompt feedback: blockReason=%s, message=%s", model, resp.PromptFeedback.BlockReason, resp.PromptFeedback.BlockReasonMessage)
		}
		if candidate != nil {
			finishReason = candidate.FinishReason
			docLogger.Errorf("[%s] Candidate: finishReason=%s, finishMessage=%s", model, candidate.FinishReason, candidate.FinishMessage)
			for _, sr := range candidate.SafetyRatings {
				if sr.Blocked {
					docLogger.Errorf("[%s] Safety blocked: category=%s, probability=%s", model, sr.Category, sr.Probability)
				}
			}
		}
		if resp.UsageMetadata != nil {
			docLogger.Infof("[%s] Usage: promptTokens=%d, candidateTokens=%d, totalTokens=%d",
				model, resp.UsageMetadata.PromptTokenCount, resp.UsageMetadata.CandidatesTokenCount, resp.UsageMetadata.TotalTokenCount)
		}
	}

	partKind := oneshotPartKind(firstPart)
	msg := fmt.Sprintf("oneshot LLM returned empty/non-text response (model=%s, input size=%d bytes, finishReason=%s, partKind=%s)", model, inputSizeBytes, finishReason, partKind)
	if permanentFinishReasons[finishReason] {
		return "", &permanentOneshotError{msg: msg}
	}
	return "", fmt.Errorf("%s", msg)
}

func detectOneshotInputMIMEType(inputBytes []byte) (string, error) {
	if len(inputBytes) == 0 {
		return "", fmt.Errorf("oneshot input is empty")
	}

	detectedMIME := mimetype.Detect(inputBytes).String()
	if detectedMIME == "application/pdf" || strings.HasPrefix(detectedMIME, "image/") {
		return detectedMIME, nil
	}

	return "", fmt.Errorf("unsupported oneshot document MIME type: %s (expected application/pdf or image/*)", detectedMIME)
}

// callOneshotModel sends a document + prompt to the given model and returns the response text.
// Returns a permanentOneshotError for non-retryable failures.
func callOneshotModel(ctx context.Context, client *genai.Client, model string, mimeType string, inputBytes []byte, prompt string, docLogger *logrus.Entry) (string, error) {
	resp, err := client.Models.GenerateContent(ctx, model, []*genai.Content{
		{
			Parts: []*genai.Part{
				{InlineData: &genai.Blob{
					MIMEType: mimeType,
					Data:     inputBytes,
				}},
				{Text: prompt},
			},
			Role: "user",
		},
	}, nil)
	if err != nil {
		return "", fmt.Errorf("oneshot LLM call failed (model=%s): %w", model, err)
	}
	return extractOneshotTextFromResponse(resp, model, len(inputBytes), docLogger)
}

// generateOneshotSuggestion processes a single document using the oneshot multimodal approach.
// It sends the raw document bytes (PDF or image) to Google AI Gemini along with a structured prompt, and parses the response.
func (app *App) generateOneshotSuggestion(
	ctx context.Context,
	doc Document,
	req GenerateSuggestionsRequest,
	availableTagNames []string,
	availableCorrespondentNames []string,
	availableDocumentTypeNames []string,
) (*DocumentSuggestion, error) {
	docLogger := documentLogger(doc.ID)

	// Download document in its original format for oneshot processing.
	docLogger.Info("Downloading document for oneshot processing")
	_, inputBytes, _, err := app.Client.DownloadDocumentAsPDF(ctx, doc.ID, limitOcrPages, false)
	if err != nil {
		return nil, fmt.Errorf("failed to download document %d for oneshot: %w", doc.ID, err)
	}

	if len(inputBytes) == 0 {
		return nil, fmt.Errorf("downloaded input for document %d is empty", doc.ID)
	}

	inputMIMEType, err := detectOneshotInputMIMEType(inputBytes)
	if err != nil {
		return nil, fmt.Errorf("failed to detect oneshot MIME type for document %d: %w", doc.ID, err)
	}

	docLogger.Infof("Downloaded document input (%d bytes, mime=%s), sending to %s for OCR + field extraction", len(inputBytes), inputMIMEType, oneshotModel)

	// Build custom fields XML if needed
	var customFieldsXML string
	var selectedCustomFields []CustomField
	if req.GenerateCustomFields {
		settingsMutex.RLock()
		selectedIDs := settings.CustomFieldsSelectedIDs
		settingsMutex.RUnlock()

		if len(selectedIDs) > 0 {
			allCustomFields, err := app.Client.GetCustomFields(ctx)
			if err != nil {
				return nil, fmt.Errorf("error fetching custom fields: %w", err)
			}

			for _, field := range allCustomFields {
				for _, selectedID := range selectedIDs {
					if field.ID == selectedID {
						selectedCustomFields = append(selectedCustomFields, field)
						break
					}
				}
			}

			if len(selectedCustomFields) > 0 {
				var xmlBuilder strings.Builder
				xmlBuilder.WriteString("<custom_fields>\n")
				for _, field := range selectedCustomFields {
					escapedName := html.EscapeString(field.Name)
					escapedType := html.EscapeString(field.DataType)
					xmlBuilder.WriteString("  <field name=\"")
					xmlBuilder.WriteString(escapedName)
					xmlBuilder.WriteString("\" type=\"")
					xmlBuilder.WriteString(escapedType)
					xmlBuilder.WriteString("\"></field>\n")
				}
				xmlBuilder.WriteString("</custom_fields>")
				customFieldsXML = xmlBuilder.String()
			}
		}
	}

	// Build prompt from template
	templateMutex.RLock()
	tmpl := oneshotTemplate
	templateMutex.RUnlock()

	// Clean available tags (remove system tags)
	cleanedTags := removeSystemTagsFromList(availableTagNames)

	templateData := map[string]interface{}{
		"Language":                getLikelyLanguage(),
		"Today":                   getTodayDate(),
		"OriginalTitle":           doc.Title,
		"OriginalTags":            doc.Tags,
		"GenerateTitle":           req.GenerateTitles,
		"GenerateTags":            req.GenerateTags,
		"GenerateCorrespondent":   req.GenerateCorrespondents,
		"GenerateDocumentType":    req.GenerateDocumentTypes,
		"GenerateCreatedDate":     req.GenerateCreatedDate,
		"GenerateCustomFields":    req.GenerateCustomFields && len(selectedCustomFields) > 0,
		"AvailableTags":           cleanedTags,
		"AvailableCorrespondents": availableCorrespondentNames,
		"CorrespondentBlackList":  correspondentBlackList,
		"AvailableDocumentTypes":  availableDocumentTypeNames,
		"CustomFieldsXML":         customFieldsXML,
	}

	var promptBuffer bytes.Buffer
	if err := tmpl.Execute(&promptBuffer, templateData); err != nil {
		return nil, fmt.Errorf("error executing oneshot template: %w", err)
	}
	prompt := promptBuffer.String()
	docLogger.Debugf("Oneshot prompt: %s", prompt)

	// Get or create the Google AI client
	client, err := getOrCreateOneshotLLM()
	if err != nil {
		return nil, fmt.Errorf("failed to get oneshot LLM: %w", err)
	}

	// Try primary model
	responseText, err := callOneshotModel(ctx, client, oneshotModel, inputMIMEType, inputBytes, prompt, docLogger)
	if err != nil {
		// If permanent failure and backup model is configured, try backup
		if isPermanentOneshotError(err) && oneshotBackupModel != "" {
			docLogger.Warnf("Primary model %s failed permanently, trying backup model %s", oneshotModel, oneshotBackupModel)
			responseText, err = callOneshotModel(ctx, client, oneshotBackupModel, inputMIMEType, inputBytes, prompt, docLogger)
		}
		if err != nil {
			return nil, err
		}
	}
	docLogger.Debugf("Oneshot raw response: %s", responseText)

	// Strip reasoning and markdown
	responseText = stripReasoning(responseText)
	responseText = stripMarkdown(responseText)

	// Extract JSON object - LLMs sometimes wrap the response in extra quotes or text
	if start := strings.Index(responseText, "{"); start >= 0 {
		if end := strings.LastIndex(responseText, "}"); end >= start {
			responseText = responseText[start : end+1]
		}
	}

	// Parse JSON response
	var oneshotResp oneshotResponse
	if err := json.Unmarshal([]byte(responseText), &oneshotResp); err != nil {
		return nil, fmt.Errorf("failed to parse oneshot response for document %d: %w (response: %s)", doc.ID, err, responseText)
	}

	// Build suggestion with validation
	suggestion := DocumentSuggestion{
		ID:               doc.ID,
		OriginalDocument: doc,
		RemoveTags:       []string{autoOneshotTag, autoTag},
	}

	// Content (OCR text)
	if oneshotResp.Content != "" {
		suggestion.SuggestedContent = oneshotResp.Content
		docLogger.Infof("Oneshot extracted %d characters of OCR content", len(oneshotResp.Content))
	}

	// Title
	if req.GenerateTitles && oneshotResp.Title != "" {
		suggestion.SuggestedTitle = strings.TrimSpace(strings.Trim(oneshotResp.Title, "\""))
	} else {
		suggestion.SuggestedTitle = doc.Title
	}

	// Tags - filter against available tags
	if req.GenerateTags {
		filteredTags := filterTagsAgainstAvailable(oneshotResp.Tags, cleanedTags)
		// Append original tags and deduplicate
		filteredTags = append(filteredTags, doc.Tags...)
		slices.Sort(filteredTags)
		filteredTags = slices.Compact(filteredTags)
		suggestion.SuggestedTags = filteredTags
	} else {
		suggestion.SuggestedTags = doc.Tags
	}

	// Correspondent
	if req.GenerateCorrespondents {
		suggestion.SuggestedCorrespondent = strings.TrimSpace(oneshotResp.Correspondent)
	}

	// Document Type - validate against available types
	if req.GenerateDocumentTypes {
		suggestion.SuggestedDocumentType = validateDocumentType(oneshotResp.DocumentType, availableDocumentTypeNames)
	}

	// Created Date
	if req.GenerateCreatedDate {
		suggestion.SuggestedCreatedDate = strings.TrimSpace(oneshotResp.CreatedDate)
	}

	// Custom Fields
	if req.GenerateCustomFields && len(selectedCustomFields) > 0 {
		suggestion.SuggestedCustomFields = validateCustomFields(oneshotResp.CustomFields, selectedCustomFields)
	}

	// Apply settings
	settingsMutex.RLock()
	suggestion.CustomFieldsWriteMode = settings.CustomFieldsWriteMode
	suggestion.CustomFieldsEnable = settings.CustomFieldsEnable
	settingsMutex.RUnlock()

	return &suggestion, nil
}

// filterTagsAgainstAvailable filters suggested tags to only include those in the available list (case-insensitive)
func filterTagsAgainstAvailable(suggested []string, available []string) []string {
	filtered := make([]string, 0, len(suggested))
	for _, tag := range suggested {
		tag = strings.TrimSpace(tag)
		for _, availableTag := range available {
			if strings.EqualFold(tag, availableTag) {
				filtered = append(filtered, availableTag)
				break
			}
		}
	}
	return filtered
}

// validateDocumentType checks if the suggested type is in the available list (case-insensitive)
func validateDocumentType(suggested string, available []string) string {
	suggested = strings.TrimSpace(suggested)
	for _, docType := range available {
		if strings.EqualFold(suggested, docType) {
			return docType
		}
	}
	if suggested != "" {
		log.Warnf("Oneshot suggested document type '%s' not found in available types, ignoring", suggested)
	}
	return ""
}

// validateCustomFields maps oneshot custom field results back to known field IDs
func validateCustomFields(suggested []struct {
	Field string      `json:"field"`
	Value interface{} `json:"value"`
}, available []CustomField) []CustomFieldSuggestion {
	fieldNameIDMap := make(map[string]int)
	for _, field := range available {
		fieldNameIDMap[field.Name] = field.ID
	}

	var result []CustomFieldSuggestion
	for _, s := range suggested {
		if id, ok := fieldNameIDMap[s.Field]; ok {
			result = append(result, CustomFieldSuggestion{
				ID:    id,
				Name:  s.Field,
				Value: s.Value,
			})
		} else {
			log.Warnf("Oneshot returned unknown custom field '%s', skipping", s.Field)
		}
	}
	return result
}
