import ai.djl.Device;
import ai.djl.MalformedModelException;
import ai.djl.inference.Predictor;
import ai.djl.modality.cv.Image;
import ai.djl.modality.cv.ImageFactory;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ModelNotFoundException;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.TranslateException;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class SupResTest {
    public static void main(String[] args) throws IOException, ModelNotFoundException, MalformedModelException {
        Path modelpath = Paths.get("src/main/trained_models/superResolution/");
        Path imgpath = Paths.get("src/main/saveimg/comic.png");
        System.out.println(imgpath);
        Image image = ImageFactory.getInstance().fromFile(imgpath);

        Criteria<Image, Image> criteria = Criteria.builder()
                .setTypes(Image.class, Image.class)
                .optModelPath(modelpath)
                .optModelName("RRDB_ESRGAN_x4.pt")
                .optTranslator(new superResolutionTranslator())
                .optProgress(new ProgressBar())
                .optDevice(Device.gpu())
                .build();

        System.out.println("loadModel");
        try (ZooModel<Image, Image> zooModel = criteria.loadModel()){
            System.out.println("Finish");
            try (Predictor<Image, Image> predictor = zooModel.newPredictor()){
                System.out.println("Predictor");
                Image supImage = predictor.predict(image);
                System.out.println("Finish");
                Path output = Paths.get("src\\main\\saveimg\\output");
                Path resolve = output.resolve(output.getNameCount() + ".png");
                System.out.println("Saving");
                supImage.save(Files.newOutputStream(resolve), "png");
                System.out.println("Finish");
            } catch (TranslateException e) {
                throw new RuntimeException(e);
            }
        }
    }
}
