package main

import (
	"context"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
)

type cycleStub struct {
	ocrEnabled     bool
	oneshotEnabled bool

	ocrCount     int
	autoCount    int
	oneshotCount int

	ocrErr     error
	autoErr    error
	oneshotErr error

	ocrCalls     int
	autoCalls    int
	oneshotCalls int
}

func (s *cycleStub) processAutoOcrTagDocuments(ctx context.Context) (int, error) {
	s.ocrCalls++
	return s.ocrCount, s.ocrErr
}

func (s *cycleStub) processAutoTagDocuments(ctx context.Context) (int, error) {
	s.autoCalls++
	return s.autoCount, s.autoErr
}

func (s *cycleStub) processAutoOneshotDocuments(ctx context.Context) (int, error) {
	s.oneshotCalls++
	return s.oneshotCount, s.oneshotErr
}

func (s *cycleStub) isOcrEnabled() bool {
	return s.ocrEnabled
}

func (s *cycleStub) isOneshotEnabled() bool {
	return s.oneshotEnabled
}

func TestRunBackgroundCycle_OneshotRunsWhenAutoTagFails(t *testing.T) {
	stub := &cycleStub{
		ocrEnabled:     false,
		oneshotEnabled: true,
		autoErr:        errors.New("auto-tag failure"),
		oneshotCount:   2,
	}

	count, err := runBackgroundCycle(context.Background(), stub)

	assert.Equal(t, 2, count)
	assert.Error(t, err)
	assert.Equal(t, 1, stub.autoCalls)
	assert.Equal(t, 1, stub.oneshotCalls)
	assert.Contains(t, err.Error(), "processAutoTagDocuments")
}

func TestRunBackgroundCycle_AggregatesCountsAndErrors(t *testing.T) {
	stub := &cycleStub{
		ocrEnabled:     true,
		oneshotEnabled: true,
		ocrCount:       1,
		autoCount:      2,
		oneshotCount:   3,
		ocrErr:         errors.New("ocr failure"),
		oneshotErr:     errors.New("oneshot failure"),
	}

	count, err := runBackgroundCycle(context.Background(), stub)

	assert.Equal(t, 6, count)
	assert.Error(t, err)
	assert.Equal(t, 1, stub.ocrCalls)
	assert.Equal(t, 1, stub.autoCalls)
	assert.Equal(t, 1, stub.oneshotCalls)
	assert.Contains(t, err.Error(), "processAutoOcrTagDocuments")
	assert.Contains(t, err.Error(), "processAutoOneshotDocuments")
}
