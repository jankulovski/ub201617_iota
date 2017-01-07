/*
* For usage in envirement variables add path to libraries:
* jcommon-1.0.23.jar
* jfreechart-1.0.19.jar
* json-simple-1.1.1.jar
* */
public class Main {

    public static void main(String[] args) {
        Chart demo = new Chart("Analyze");
        demo.pack();
        demo.setVisible(true);
    }
}
