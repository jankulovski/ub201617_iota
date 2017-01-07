/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package agentsanalyzegui;
import javax.swing.JFrame;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.ChartPanel;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import java.awt.Color;
import java.awt.BasicStroke;
import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.io.*;
import org.jfree.chart.plot.XYPlot;
import org.jfree.chart.renderer.xy.XYLineAndShapeRenderer;
import org.jfree.data.xy.XYDataset;
import org.jfree.data.xy.XYSeries;
import org.jfree.data.xy.XYSeriesCollection;
import org.json.JSONException;
import org.json.JSONObject;
import org.json.JSONArray;


public class Chart extends JFrame {

    private static final long serialVersionUID = 1L;
    public static String chartTitle = "";

    public Chart(String applicationTitle) {
        super(applicationTitle);
        XYDataset dataset;
        try{
            dataset = createDataset();
        }
        catch (Exception e){
            dataset = new XYSeriesCollection();
        }
        JFreeChart chart = createChart(dataset, chartTitle);
        ChartPanel chartPanel = new ChartPanel(chart);
        chartPanel.setPreferredSize(new java.awt.Dimension(500, 270));
        setContentPane(chartPanel);

    }

    /**
     * Creates a sample dataset
     */
    private  XYDataset  createDataset() throws JSONException {
        String file;
        try {
            file = GetFile();
        }catch (Exception e){
            file = "";
        }

        final JSONObject obj = new JSONObject(file);

        final XYSeries gener = new XYSeries( "Agents fitness" );
        if(obj != null) {
            String crossoverClassName = obj.getString("crossoverClassName");
            String selectionClassName = obj.getString("selectionClassName");
            chartTitle = "Chart using crossover: " + crossoverClassName+ " and selection: " + selectionClassName;
            for (int i = 0; i < obj.length() - 2; i++) {
                int g = i + 1;
//                if(g == 0)
//                    g = 1;
                final JSONArray data = obj.getJSONObject(Integer.toString(g)).getJSONArray("agents");
                if (data.length() > 0) {
                    final JSONObject m = data.getJSONObject(0);
                    int fitness = m.getInt("fitness");
                    gener.add(i + 1, fitness);
                }
            }
        }
        final XYSeriesCollection dataset = new XYSeriesCollection( );
        dataset.addSeries( gener );
        return dataset;

    }

    private JFreeChart createChart(XYDataset dataset, String title) {

        JFreeChart chart = ChartFactory.createXYLineChart(
                title,
                "Generations", "Fitness",
                dataset,
                PlotOrientation.VERTICAL,
                true,
                true,
                false
        );
        final XYPlot plot = chart.getXYPlot( );
        XYLineAndShapeRenderer renderer = new XYLineAndShapeRenderer( );
        renderer.setSeriesPaint( 0 , Color.RED );
        renderer.setSeriesPaint( 1 , Color.GREEN );
        renderer.setSeriesPaint( 2 , Color.YELLOW );
        renderer.setSeriesStroke( 0 , new BasicStroke( 1.0f ) );
        renderer.setSeriesStroke( 1 , new BasicStroke( 3.0f ) );
        renderer.setSeriesStroke( 2 , new BasicStroke( 2.0f ) );
        plot.setRenderer( renderer );
        return chart;
    }

    private String GetFile() throws Exception{
        String path = System.getProperty("user.dir");
        path = path.replace(File.separator + "AgentsAnalyzeGUI","");
        path += File.separator + "output" + File.separator + "output.txt";
        FileReader file = new FileReader(path);
        BufferedReader textReader = new BufferedReader(file);
        String jsonString = textReader.readLine();
        return jsonString;
    }

}