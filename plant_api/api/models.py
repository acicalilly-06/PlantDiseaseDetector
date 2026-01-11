from django.db import models

class PredictionRecord(models.Model):
    image = models.URLField()
    predicted_label = models.CharField(max_length=255)
    confidence = models.FloatField()
    model_type = models.CharField(max_length=50, default="CNN")

    cloudinary_original_id = models.CharField(max_length=255, null=True, blank=True)
    cloudinary_preprocessed_id = models.CharField(max_length=255, null=True, blank=True)
    cloudinary_segmented_id = models.CharField(max_length=255, null=True, blank=True)

    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.predicted_label} ({self.confidence:.2f})"
