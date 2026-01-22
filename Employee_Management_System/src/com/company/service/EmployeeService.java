package com.company.service;

import java.util.List;
import com.company.dao.EmployeeDAO;
import com.company.dao.EmployeeDAOImpl;
import com.company.dto.EmployeeDTO;

public class EmployeeService {

    private EmployeeDAO dao = new EmployeeDAOImpl();

    public void addEmployee(EmployeeDTO emp) {
        if (emp.getEmpId() <= 0) {
            System.out.println("Invalid Employee ID!");
            return;
        }

        if (emp.getSalary() <= 0) {
            System.out.println("Salary must be positive!");
            return;
        }

        dao.addEmployee(emp);
    }

    public List<EmployeeDTO> getAllEmployees() {
        return dao.getAllEmployees();
    }

    public EmployeeDTO getEmployeeById(int id) {
        return dao.getEmployeeById(id);
    }

    public void updateSalary(int id, double salary) {
        if (salary <= 0) {
            System.out.println("Invalid salary!");
            return;
        }
        dao.updateSalary(id, salary);
    }

    public void deleteEmployee(int id) {
        dao.deleteEmployee(id);
    }
}
