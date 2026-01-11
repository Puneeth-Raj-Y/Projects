package EmployeeRA;

import javafx.application.Application;
import javafx.concurrent.Worker;
import javafx.scene.Scene;
import javafx.scene.web.WebView;
import javafx.stage.Stage;
import netscape.javascript.JSObject;
import java.util.*;

public class EmployeeWebApp extends Application {

    static class Employee {
        int id, k1, k2, k3;
        String name;
        double avg;
        String grade;
    }

    List<Employee> list = new ArrayList<>();
    JSObject window;

    @Override
    public void start(Stage stage) {
        WebView web = new WebView();

        web.getEngine().load(
            Objects.requireNonNull(
                getClass().getResource("/EmployeeRA/ui.html")
            ).toExternalForm()
        );

        web.getEngine().getLoadWorker().stateProperty().addListener((obs, old, state) -> {
            if (state == Worker.State.SUCCEEDED) {
                window = (JSObject) web.getEngine().executeScript("window");
                window.setMember("java", this);
            }
        });

        stage.setScene(new Scene(web, 1100, 650));
        stage.setTitle("Employee Performance System");
        stage.show();
    }
    void validate(String id, String k1, String k2, String k3){
        // ID must be letters + numbers
        if(!id.matches("[A-Za-z]+[0-9]+")){
            window.call("showError","ID must contain letters followed by numbers (ex: EMP101)");
            throw new RuntimeException();
        }

        // KPI must be numbers only
        if(!k1.matches("\\d+") || !k2.matches("\\d+") || !k3.matches("\\d+")){
            window.call("showError","Marks must contain only numbers");
            throw new RuntimeException();
        }

        int a=Integer.parseInt(k1), b=Integer.parseInt(k2), c=Integer.parseInt(k3);

        // KPI range
        if(a>100||b>100||c>100){
            window.call("showError","Marks must be between 0 and 100");
            throw new RuntimeException();
        }
    }

    // ================= ADD =================
    public void add(String id, String name, String k1, String k2, String k3) {
    	validate(id,k1,k2,k3);
    	int eid = Integer.parseInt(id.replaceAll("[^0-9]",""));


    	for(Employee emp:list){
    	    if(emp.id == eid){
    	        window.call("showError","Employee ID already exists!");
    	        return;
    	    }
    	}


        Employee e = new Employee();
        e.id = eid;
        e.name = name;
        e.k1 = Integer.parseInt(k1);
        e.k2 = Integer.parseInt(k2);
        e.k3 = Integer.parseInt(k3);
        calc(e);
        list.add(e);
        send();
        window.call("showSuccess", "Employee added successfully!");
        window.call("onAddSuccess");
    }

    // ================= UPDATE =================
    public void update(String id, String name, String k1, String k2, String k3) {
    	validate(id,k1,k2,k3);
    	int uid = Integer.parseInt(id.replaceAll("[^0-9]",""));
        boolean found = false;

        for (Employee e : list) {
            if (e.id == uid) {
                e.name = name;
                e.k1 = Integer.parseInt(k1);
                e.k2 = Integer.parseInt(k2);
                e.k3 = Integer.parseInt(k3);
                calc(e);
                found = true;
                break;
            }
        }

        if (found) {
            send();
            window.call("showSuccess", "Employee updated successfully!");
        } else {
            window.call("showError", "Employee ID not found!");
        }
        window.call("onManageSuccess");
    }

    // ================= DELETE =================
    public void remove(String id) {
    	if(!id.matches("[A-Za-z]+[0-9]+")){
    	    window.call("showError","Invalid ID format");
    	    return;
    	}
    	int uid = Integer.parseInt(id.replaceAll("[^0-9]",""));
        boolean removed = list.removeIf(e -> e.id == uid);

        if (removed) {
            send();
            window.call("showSuccess", "Employee deleted successfully!");
        } else {
            window.call("showError", "Employee ID not found!");
        }
        window.call("onManageSuccess");
    }

    // ================= SORT / REFRESH =================
    public void sort() {
        list.sort(Comparator.comparingInt(e -> gradeRank(e.grade)));
        send();
    }

    public void refresh() {
        list.sort(Comparator.comparingInt(e -> e.id));
        send();
    }

    // ================= SEARCH =================
    public void search(String id){

        // Validate ID format
        if(!id.matches("[A-Za-z]+[0-9]+")){
            window.call("showError", "Invalid ID format. Use like EMP101");
            return;
        }

        int sid = Integer.parseInt(id.replaceAll("[^0-9]",""));
        boolean found = false;

        StringBuilder sb = new StringBuilder();

        for(Employee e : list){
            if(e.id == sid){
                sb.append(e.id).append("|")
                  .append(e.name).append("|")
                  .append(String.format("%.2f", e.avg)).append("|")
                  .append(e.grade).append(";");
                found = true;
                break;
            }
        }

        window.call("updateSearchTable", sb.toString());

        if(found){
            window.call("showSuccess", "Employee found successfully!");
        } else {
            window.call("showError", "Employee not found!");
        }
    }



    // ================= LOGIC =================
    void calc(Employee e) {
        e.avg = (e.k1 + e.k2 + e.k3) / 3.0;
        if (e.avg >= 85) e.grade = "A";
        else if (e.avg >= 70) e.grade = "B";
        else if (e.avg >= 50) e.grade = "C";
        else e.grade = "F";
    }

    int gradeRank(String g) {
        switch (g) {
            case "A": return 1;
            case "B": return 2;
            case "C": return 3;
            default: return 4;
        }
    }

    void send() {
        StringBuilder sb = new StringBuilder();
        for (Employee e : list) {
            sb.append(e.id).append("|")
              .append(e.name).append("|")
              .append(String.format("%.2f", e.avg)).append("|")
              .append(e.grade).append(";");
        }
        window.call("updateTable", sb.toString());
    }

    public static void main(String[] args) {
        launch();
    }
}
