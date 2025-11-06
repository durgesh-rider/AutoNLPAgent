from typing import Dict, Any, Optional
from app.core.task_detector import detect_task_type, analyze_dataset_structure
from app.core.preprocessor import TextPreprocessor
from app.models.response import TaskDetectionResponse, PreprocessingResponse


class NLPService:
    def __init__(self):
        self.preprocessor = TextPreprocessor()

    def detect_task(self, df, dataset_id: str) -> TaskDetectionResponse:
        """Detect NLP task type from dataset"""
        try:
            task_type = detect_task_type(df)
            confidence = self._calculate_detection_confidence(df, task_type)

            return TaskDetectionResponse(
                success=True,
                message="Task detection completed",
                dataset_id=dataset_id,
                detected_task=task_type,
                confidence=confidence,
                reasoning=self._get_detection_reasoning(df, task_type)
            )
        except Exception as e:
            return TaskDetectionResponse(
                success=False,
                message=f"Task detection failed: {str(e)}",
                dataset_id=dataset_id,
                detected_task="unknown",
                confidence=0.0,
                reasoning="Error during detection"
            )

    def preprocess_data(self, df, dataset_id: str, task_type: str) -> PreprocessingResponse:
        """Preprocess dataset for NLP tasks"""
        try:
            processed_df, metadata = self.preprocessor.preprocess_dataset(df, task_type)

            return PreprocessingResponse(
                success=True,
                message="Data preprocessing completed",
                dataset_id=dataset_id,
                preprocessing_steps=metadata['preprocessing_steps'],
                processed_data_shape=metadata['processed_shape'],
                missing_values_handled=metadata['original_shape'][0] * metadata['original_shape'][1] - processed_df.notna().sum().sum(),
                text_cleaned=len(metadata.get('text_columns', [])) > 0
            )
        except Exception as e:
            return PreprocessingResponse(
                success=False,
                message=f"Preprocessing failed: {str(e)}",
                dataset_id=dataset_id,
                preprocessing_steps=[],
                processed_data_shape=(0, 0),
                missing_values_handled=0,
                text_cleaned=False
            )

    def _calculate_detection_confidence(self, df, task_type: str) -> float:
        """Calculate confidence score for task detection"""
        if task_type == "unknown":
            return 0.0

        # Simple confidence calculation based on data patterns
        confidence = 0.5  # Base confidence

        # Increase confidence based on clear indicators
        columns = df.columns.tolist()

        if task_type == "sentiment_analysis":
            sentiment_indicators = ['sentiment', 'polarity', 'emotion']
            if any(indicator in ' '.join(columns).lower() for indicator in sentiment_indicators):
                confidence += 0.3

        elif task_type == "classification":
            if len(df.select_dtypes(include=['object']).columns) >= 2:
                confidence += 0.2

        # Check data quality
        if df.isnull().sum().sum() == 0:
            confidence += 0.1

        return min(confidence, 1.0)

    def _get_detection_reasoning(self, df, task_type: str) -> str:
        """Provide reasoning for task detection"""
        if task_type == "sentiment_analysis":
            return "Detected sentiment-related column names or content patterns"
        elif task_type == "classification":
            return "Found text column with corresponding label column"
        elif task_type == "ner":
            return "Detected entity tagging patterns in data"
        elif task_type == "question_answering":
            return "Found question-answer pattern in columns"
        elif task_type == "summarization":
            return "Detected summary-related column names"
        else:
            return "Could not determine specific NLP task from data structure"


# Global NLP service instance
nlp_service = NLPService()