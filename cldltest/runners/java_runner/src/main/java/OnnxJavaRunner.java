import ai.onnxruntime.*;
import java.nio.FloatBuffer;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Map;

public class OnnxJavaRunner {
    public static void main(String[] args) throws Exception {
        if (args.length < 3) {
            System.out.println("Usage: java OnnxJavaRunner <modelPath> <inputJsonPath> <outputJsonPath>");
            System.exit(1);
        }

        String modelPath = args[0];
        String inputJsonPath = args[1];
        String outputJsonPath = args[2];

        String json = Files.readString(Paths.get(inputJsonPath));

        String inputName = extractString(json, "input_name");
        float[] data = extractFloatArray(json, "data");
        long[] shape = extractLongArray(json, "shape");

        OrtEnvironment env = OrtEnvironment.getEnvironment();
        OrtSession.SessionOptions sessionOptions = new OrtSession.SessionOptions();
        OrtSession session = env.createSession(modelPath, sessionOptions);

        OnnxTensor tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(data), shape);
        Map<String, OnnxTensor> inputs = Map.of(inputName, tensor);

        float[] outputData;

        try (OrtSession.Result results = session.run(inputs)) {
            OnnxTensor outputTensor = (OnnxTensor) results.get(0);
            float[][] arr = (float[][]) outputTensor.getValue();

            outputData = new float[arr.length];
            for (int i = 0; i < arr.length; i++) {
                outputData[i] = arr[i][0];
            }
        }

        String resultJson =
                "{\n" +
                "  \"runner\": \"java_onnxruntime\",\n" +
                "  \"model\": \"" + escape(modelPath) + "\",\n" +
                "  \"input_name\": \"" + escape(inputName) + "\",\n" +
                "  \"input_shape\": [" + shape[0] + ", " + shape[1] + "],\n" +
                "  \"input_data\": " + arrayToJson(data) + ",\n" +
                "  \"output_shape\": [" + outputData.length + ", 1],\n" +
                "  \"output_data\": " + arrayToJson(outputData) + ",\n" +
                "  \"dtype\": \"float32\",\n" +
                "  \"status\": \"success\"\n" +
                "}\n";

        Files.writeString(Paths.get(outputJsonPath), resultJson);

        tensor.close();
        session.close();
        env.close();

        System.out.println("Java backend done: " + outputJsonPath);
    }

    static String extractString(String json, String key) {
        String pattern = "\"" + key + "\":";
        int start = json.indexOf(pattern);
        int q1 = json.indexOf("\"", start + pattern.length());
        int q2 = json.indexOf("\"", q1 + 1);
        return json.substring(q1 + 1, q2);
    }

    static float[] extractFloatArray(String json, String key) {
        String pattern = "\"" + key + "\":";
        int start = json.indexOf(pattern);
        int l = json.indexOf("[", start);
        int r = json.indexOf("]", l);
        String[] parts = json.substring(l + 1, r).split(",");
        float[] arr = new float[parts.length];
        for (int i = 0; i < parts.length; i++) {
            arr[i] = Float.parseFloat(parts[i].trim());
        }
        return arr;
    }

    static long[] extractLongArray(String json, String key) {
        String pattern = "\"" + key + "\":";
        int start = json.indexOf(pattern);
        int l = json.indexOf("[", start);
        int r = json.indexOf("]", l);
        String[] parts = json.substring(l + 1, r).split(",");
        long[] arr = new long[parts.length];
        for (int i = 0; i < parts.length; i++) {
            arr[i] = Long.parseLong(parts[i].trim());
        }
        return arr;
    }

    static String arrayToJson(float[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(arr[i]);
        }
        sb.append("]");
        return sb.toString();
    }

    static String escape(String s) {
        return s.replace("\\", "\\\\").replace("\"", "\\\"");
    }
}