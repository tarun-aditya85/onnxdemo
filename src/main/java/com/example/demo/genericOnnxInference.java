package com.example.demo;


import ai.onnxruntime.*;
import java.nio.LongBuffer;
import java.util.Map;


public class genericOnnxInference {


        public static void main(String args[]) throws OrtException {
            // Load the model and create InferenceSession
            System.out.println("This is the model loading");
            String modelPath = "../../notebooks/deepfm_dnn.onnx";
            OrtEnvironment env = OrtEnvironment.getEnvironment();
            OrtSession session = env.createSession(modelPath);

            // The actual tokens (manually because I don't know where to find the tokenizer!)
            long[] inputData = new long[]{32, 1, 2, 35};  // Your input data in the appropriate format
            long[] inputShape = new long[]{1, 4};  // Shape of the input data

            // Attention mask needed (all ones - same shape as the input ids)
            long[] inputData_a = new long[]{1, 1, 1, 1};  // Your input data in the appropriate format
            long[] inputShape_a = new long[]{1, 4};  // Shape of the input data

            OnnxTensor inputTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputData), inputShape);
            OnnxTensor attTensor = OnnxTensor.createTensor(env, LongBuffer.wrap(inputData_a), inputShape_a);

            // This is the forward pass so that we can get the logits
            var inputs = Map.of("input_ids", inputTensor, "attention_mask", attTensor);
            var results = session.run(inputs);

            // This is the output (one of the outputs)
            OnnxValue a = results.get(0);
            System.out.println(a.getValue());
            System.out.println(a.getInfo());
            System.out.println(a.getInfo());

        }


}
