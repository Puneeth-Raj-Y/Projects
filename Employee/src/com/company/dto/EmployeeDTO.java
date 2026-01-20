package com.company.dto;

public class EmployeeDTO {

    private int empId;
    private String name;
    private String department;
    private double salary;
    private String city;

    public EmployeeDTO() {}

    public EmployeeDTO(int empId, String name, String department, double salary, String city) {
        this.empId = empId;
        this.name = name;
        this.department = department;
        this.salary = salary;
        this.city = city;
    }

    public int getEmpId() { return empId; }
    public void setEmpId(int empId) { this.empId = empId; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    public String getDepartment() { return department; }
    public void setDepartment(String department) { this.department = department; }

    public double getSalary() { return salary; }
    public void setSalary(double salary) { this.salary = salary; }

    public String getCity() { return city; }
    public void setCity(String city) { this.city = city; }
}
