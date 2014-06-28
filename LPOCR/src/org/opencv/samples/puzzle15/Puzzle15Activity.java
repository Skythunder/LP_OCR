package org.opencv.samples.puzzle15;

import java.util.ArrayList;
import java.util.List;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfPoint;
import org.opencv.core.MatOfPoint2f;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.highgui.Highgui;
import org.opencv.imgproc.Imgproc;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener;
import org.opencv.android.JavaCameraView;

import com.googlecode.tesseract.android.TessBaseAPI;

import android.os.Bundle;
import android.os.Environment;
import android.app.Activity;
import android.graphics.Bitmap;
import android.util.Log;
import android.view.Menu;
import android.view.MenuItem;
import android.view.MotionEvent;
import android.view.View;
import android.view.WindowManager;
import android.view.View.OnClickListener;
import android.widget.Button;
import android.widget.EditText;

public class Puzzle15Activity extends Activity {

    private static final String  TAG = "HELLO!";

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {

        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    Log.i(TAG, "OpenCV loaded successfully");

                    
                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.layout);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        Log.d(TAG, "Creating and seting view");
        Button b=(Button)findViewById(R.id.button1);
        b.setOnClickListener(new OnClickListener(){
            @Override
            public void onClick(View v) {
                     EditText t = (EditText)findViewById(R.id.txt1);
                     String s = t.getText().toString();
                     lpocr(s);
                     //t.setText(Environment.getExternalStorageDirectory().getAbsolutePath());
            	Log.i( TAG, "wawawawawa" );

            }
        });
        Button b2=(Button)findViewById(R.id.button2);
        b2.setOnClickListener(new OnClickListener(){
            @Override
            public void onClick(View v) {
                     finish();
                     System.exit(0);

            }
        });
    }

    @Override
    public void onPause()
    {
        super.onPause();

    }

    @Override
    public void onResume()
    {
        super.onResume();
        OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_3, this, mLoaderCallback);
    }

    public void onDestroy() {
        super.onDestroy();

    }

    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        Log.i(TAG, "called onCreateOptionsMenu");
        
        return true;
    }

    @Override
    public boolean onOptionsItemSelected(MenuItem item) {
        Log.i(TAG, "Menu Item selected " + item);
        return true;
    }

    public void lpocr(String filename)
    {
    	//Mat img=Highgui.imread("/storage/extSdCard/DCIM/Camera/"+filename);
    	
    	Mat img=Highgui.imread(Environment.getExternalStorageDirectory().getPath()+filename);
    	//Mat img=Highgui.imread("/storage/extSdCard/test02.jpg");
    	EditText t = (EditText)findViewById(R.id.txt1);
    	if(img.empty()){
    		t.setText("Image not found!");
    		return;
    	}
    	Mat gray = new Mat(img.size(),CvType.CV_8U);
    	Imgproc.cvtColor(img,gray,Imgproc.COLOR_BGR2GRAY);
    	Imgproc.blur(gray, gray, new Size(3,3));
    	Mat bin=new Mat(img.size(),CvType.CV_8U);
    	Imgproc.threshold(gray,bin,120,255,Imgproc.THRESH_BINARY);
    	List<MatOfPoint> contours = new ArrayList<MatOfPoint>();    
    	Imgproc.findContours(bin,contours,new Mat(),Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);
    	List<MatOfPoint> contours_poly = new ArrayList<MatOfPoint>();
    	List<Rect> boundRect = new ArrayList<Rect>();
    	for(int i=0;i<contours.size();i++)
    	{
    		MatOfPoint2f mMOP2f1 = new MatOfPoint2f();
    		MatOfPoint2f mMOP2f2 = new MatOfPoint2f();
    		MatOfPoint mop = new MatOfPoint();
    		contours.get(i).convertTo(mMOP2f1, CvType.CV_32FC2);   
    		Imgproc.approxPolyDP( mMOP2f1, mMOP2f2, 3, true );
    		mMOP2f2.convertTo(mop, CvType.CV_32S); 
    		contours_poly.add(mop);
    		boundRect.add(Imgproc.boundingRect( contours_poly.get(i) ));
    	}
    	List<Mat> imagens = new ArrayList<Mat>();
    	Mat drawing =Mat.zeros( img.size(), CvType.CV_8UC3 );
    	Scalar color = new Scalar( 255, 255, 255 );
    	for( int i = 0; i< contours.size(); i++ )
    	{
    		if(contours_poly.get(i).toList().size()>=4&&contours_poly.get(i).toList().size()<=20&&Imgproc.contourArea(contours.get(i))>1000)
    		{
				Imgproc.drawContours(drawing, contours_poly, i, color,-1,8,new Mat(),0,new Point());
				imagens.add(img.submat(boundRect.get(i)));
    		}
    	}
    	img.copyTo(drawing,drawing);
    	Mat aux = Mat.zeros( img.size(), CvType.CV_8UC3 );
    	Mat aux2= Mat.zeros( img.size(), CvType.CV_8UC3 );
    	Mat res = Mat.zeros( img.size(), CvType.CV_8UC3 );
    	Imgproc.cvtColor(drawing,aux,Imgproc.COLOR_BGR2GRAY);
    	Imgproc.threshold(aux,aux2,120,255,Imgproc.THRESH_BINARY);
    	Imgproc.blur( aux2, aux2, new Size(2,2) );
    	res=aux2;
    	//Imgproc.cvtColor(aux2, res, Imgproc.COLOR_RGB2BGRA);
    	Bitmap bmp = Bitmap.createBitmap(res.cols(), res.rows(), Bitmap.Config.ARGB_8888);
    	Utils.matToBitmap(res, bmp);
    	
    	TessBaseAPI baseApi = new TessBaseAPI();
    	baseApi.init("/storage/extSdCard/Tesseract", "eng");
    	baseApi.setImage(bmp);
    	String s = baseApi.getUTF8Text();
    	EditText edit = (EditText)findViewById(R.id.txt2); 
    	edit.setText(s);

    	//for(int i =0;i<imagens.size();i++)
    	//{
    	//	Highgui.imshow(Integer.toString(i),imagens.get(i));
    	//}
    	//Mat ocr=aux2;
    	//tesseract::TessBaseAPI tess;
    	
    }

}
