package com.example.demo;
// OnnxInference.java
import ai.onnxruntime.*;

import java.nio.FloatBuffer;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;

public class OnnxInference {
    private final OrtEnvironment env;
    private final OrtSession session;

    public OnnxInference(String modelPath) throws OrtException {
        this.env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        this.session = env.createSession(modelPath, sessionOptions);
    }

    public float predict(float[] input) throws OrtException {
        // Create input tensor
        long[] shape = {1, input.length}; // batch_size=1, feature_count=input.length
        OnnxTensor inputTensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(input), shape);

        // Run inference
        try {
            Map<String, OnnxTensor> inputMap = Map.of(session.getInputNames().iterator().next(), inputTensor);
            OrtSession.Result results = session.run(inputMap);

            // Get output - using the first output tensor name
            String outputName = session.getOutputNames().iterator().next();
            Optional<OnnxValue> outputOptional = results.get(outputName);

            if (!outputOptional.isPresent()) {
                throw new OrtException("No output tensor found");
            }

            OnnxTensor outputTensor = (OnnxTensor) outputOptional.get();
            float[][] outputArray = (float[][]) outputTensor.getValue();
            return outputArray[0][0]; // Return single prediction value
        } finally {
            inputTensor.close();
        }
    }

    public void close() throws OrtException {
        session.close();
        env.close();
    }
}


