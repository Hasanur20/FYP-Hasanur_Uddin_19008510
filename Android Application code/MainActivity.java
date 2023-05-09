package com.example.skinlesionclassifierv1;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.media.ThumbnailUtils;
import android.net.Uri;
import android.os.Build;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.example.skinlesionclassifierv1.ml.Model;

import org.tensorflow.lite.DataType;
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Locale;

public class MainActivity extends AppCompatActivity {


    Button cameraBtn , galleryBtn ;
    ImageView imageViewBox ;
    TextView resultTv;

    int IMAGE_SIZE = 200;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);


        cameraBtn = findViewById(R.id.cameraBtn);
        galleryBtn = findViewById(R.id.galleryBtn);

        imageViewBox = findViewById(R.id.imageViewBox);
        resultTv = findViewById(R.id.resultTv);



        // Set a click listener on the camera button
        cameraBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                    if((checkSelfPermission(Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED)){
                        Intent cameraIntent = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
                        startActivityForResult(cameraIntent, 3);

                    } else{
                        requestPermissions(new String[]{Manifest.permission.CAMERA}, 100);
                    }
                }
            }
        });


        // set gallery onclicklistener
        galleryBtn.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {

                Intent galleryIntent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
                galleryIntent.addCategory(Intent.CATEGORY_OPENABLE);
                galleryIntent.setType("image/*");
                startActivityForResult(galleryIntent, 1);

            }
        });


    }

    public void classifyingImage(Bitmap image){
        try {
            Model model = Model.newInstance(getApplicationContext());

            // Creates inputs for reference.
            TensorBuffer inputFeature0 = TensorBuffer.createFixedSize(new int[]{1, 200, 200, 3}, DataType.FLOAT32);
            // Create a byte buffer to store the image data.
            ByteBuffer byteBuffer = ByteBuffer.allocateDirect(4 * IMAGE_SIZE * IMAGE_SIZE * 3);
            byteBuffer.order(ByteOrder.nativeOrder());
            // Get the pixel values from the image and store them in an integer array.
            int[] intValues = new int [IMAGE_SIZE * IMAGE_SIZE];
            image.getPixels(intValues,0,image.getWidth(), 0, 0, image.getWidth(),image.getHeight());
            int pixel = 0;
            // Iterate through each pixel in the image.
            for (int i =0; i< IMAGE_SIZE; i++){
                for (int j = 0; j<IMAGE_SIZE; j++){
                    int val = intValues[pixel++]; //
                    // Extract and normalize the red channel value and store it in the byte buffer.
                    byteBuffer.putFloat(((val >> 16) & 0xFF) * (1.f /255));
                    // Extract and normalize the green channel value and store it in the byte buffer.
                    byteBuffer.putFloat(((val >> 8) & 0xFF) * (1.f /255));
                    // Extract and normalize the blue channel value and store it in the byte buffer.
                    byteBuffer.putFloat((val & 0xFF) * (1.f /255));

                }
            }
            // Load the byte buffer into the input tensor buffer.
            inputFeature0.loadBuffer(byteBuffer);

            // Runs model inference and gets result.
            Model.Outputs outputs = model.process(inputFeature0);
            TensorBuffer outputFeature0 = outputs.getOutputFeature0AsTensorBuffer();
            // Get the confidence scores as an array of floats.
            float[] confidence = outputFeature0.getFloatArray();
            int maxPos = 0;
            float maxConfidence = 0;
            // find the class with the highest confidence score
            for (int i = 0; i < confidence.length; i++) {
                if (confidence[i] > maxConfidence){
                    maxConfidence = confidence[i];
                    maxPos = i;
                }

            }

            String [] classes = {"Benign" , "Malignant"};


            // display the predicted class and the confidence score
            resultTv.setText(String.format("%s: %s\nConfidence Scores: [%s]",
                    getString(R.string.classifying),
                    classes[maxPos],
                    formatConfidenceScores(confidence)
            ));





            // Releases model resources if no longer used.
            model.close();
        } catch (IOException e) {
            // TODO Handle the exceptionll

        }
    }
    // method to handle confidence
    private String formatConfidenceScores(float[] confidence) {
        String[] classes = {"Benign", "Malignant"};
        StringBuilder scores = new StringBuilder();

        for (int i = 0; i < confidence.length; i++) {
            scores.append(String.format(Locale.US, "%s: %.2f", classes[i], confidence[i]*100));
            if (i < confidence.length - 1) {
                scores.append("\n ");
            }
        }

        return scores.toString();
    }


    @Override

    protected void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        // Check if the result code indicates the operation was successful (RESULT_OK)
        if(resultCode == RESULT_OK){
            // If the request code is 3, the user captured an image using the camera.
            if(requestCode == 3){
                // Extract the captured image as a Bitmap from the Intent's data.
                Bitmap image = (Bitmap) data.getExtras().get("data");
                // Calculate the minimum of image width and height.
                int dimension = Math.min(image.getWidth(), image.getHeight());
                // create a thumbnail to store the image
                image = ThumbnailUtils.extractThumbnail(image,dimension, dimension);
                imageViewBox.setImageBitmap(image);
                // Scale the image to the  image dimensions for classification.
                image = Bitmap.createScaledBitmap(image,IMAGE_SIZE, IMAGE_SIZE, false);
                // Classify the processed image.
                classifyingImage(image);
            } else{
                // If the request code is not 3, the user selected an image from the gallery.
                // Get the image's Uri from the Intent's data.
                Uri dat = data.getData();
                Bitmap image = null;
                try {// Attempt to get a Bitmap of the selected image using the ContentResolver.
                    image = MediaStore.Images.Media.getBitmap(this.getContentResolver(), dat);
                } catch (IOException e) {
                    e.printStackTrace();
                }
                // Set the selected image as the source of imageViewBox.
                imageViewBox.setImageBitmap(image);

                image = Bitmap.createScaledBitmap(image,IMAGE_SIZE, IMAGE_SIZE, false);
                classifyingImage(image);
            }
        }

        super.onActivityResult(requestCode, resultCode, data);
    }
}