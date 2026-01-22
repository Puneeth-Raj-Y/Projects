package com.company.dao;

import java.util.List;
import com.company.dto.EmployeeDTO;

public interface EmployeeDAO {

    void addEmployee(EmployeeDTO emp);

    List<EmployeeDTO> getAllEmployees();

    EmployeeDTO getEmployeeById(int empId);

    void updateSalary(int empId, double newSalary);

    void deleteEmployee(int empId);
}
