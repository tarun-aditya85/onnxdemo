package com.example;
import ai.onnxruntime.OrtException;
import com.example.demo.OnnxInference;
import org.junit.jupiter.api.Test;
import static org.junit.jupiter.api.Assertions.*;


import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import org.junit.jupiter.api.*;
import static org.junit.jupiter.api.Assertions.*;



public class OnnxInferenceTest {

// OnnxInferenceTest.java



        private static OnnxInference inference;

        @BeforeAll
        static void setUp() throws OrtException {
            // Update path to match your model location
            inference = new OnnxInference("/Users/lalithamv/Desktop/Portfolio/torchServeMultiGPU/notebooks/deepfm_dnn.onnx");
        }

        @Test
        void testPrediction() throws OrtException {
            // Create sample input with the same number of features as your training data
            float[] sampleInput = new float[39]; // 13 dense + 26 sparse features
            // Fill with sample values
            Arrays.fill(sampleInput, 0.5f);

            float prediction = inference.predict(sampleInput);

            // Assert prediction is within valid range for binary classification
            assertTrue(prediction >= 0.0f && prediction <= 1.0f,
                    "Prediction should be between 0 and 1");
        }

        @Test
        void testInvalidInput() {
            // Test with wrong input size
            float[] invalidInput = new float[10];
            Arrays.fill(invalidInput, 0.5f);

            assertThrows(OrtException.class, () -> {
                inference.predict(invalidInput);
            });
        }

        @AfterAll
        static void tearDown() throws OrtException {
            if (inference != null) {
                inference.close();
            }
        }
    }

