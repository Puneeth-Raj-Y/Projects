package com.company.dao;

import java.sql.*;
import java.util.ArrayList;
import java.util.List;

import com.company.dto.EmployeeDTO;
import com.company.util.DBConnection;

public class EmployeeDAOImpl implements EmployeeDAO {

    @Override
    public void addEmployee(EmployeeDTO emp) {
        String sql = "INSERT INTO employees VALUES (?,?,?,?,?)";

        try (Connection con = DBConnection.getConnection();
             PreparedStatement ps = con.prepareStatement(sql)) {

            ps.setInt(1, emp.getEmpId());
            ps.setString(2, emp.getName());
            ps.setString(3, emp.getDepartment());
            ps.setDouble(4, emp.getSalary());
            ps.setString(5, emp.getCity());

            ps.executeUpdate();
            System.out.println("Employee added successfully!");

        } catch (SQLException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    @Override
    public List<EmployeeDTO> getAllEmployees() {
        List<EmployeeDTO> list = new ArrayList<>();
        String sql = "SELECT * FROM employees";

        try (Connection con = DBConnection.getConnection();
             Statement st = con.createStatement();
             ResultSet rs = st.executeQuery(sql)) {

            while (rs.next()) {
                EmployeeDTO emp = new EmployeeDTO(
                        rs.getInt("emp_id"),
                        rs.getString("name"),
                        rs.getString("department"),
                        rs.getDouble("salary"),
                        rs.getString("city")
                );
                list.add(emp);
            }

        } catch (SQLException e) {
            System.out.println("Error: " + e.getMessage());
        }

        return list;
    }

    @Override
    public EmployeeDTO getEmployeeById(int empId) {
        String sql = "SELECT * FROM employees WHERE emp_id=?";

        try (Connection con = DBConnection.getConnection();
             PreparedStatement ps = con.prepareStatement(sql)) {

            ps.setInt(1, empId);
            ResultSet rs = ps.executeQuery();

            if (rs.next()) {
                return new EmployeeDTO(
                        rs.getInt("emp_id"),
                        rs.getString("name"),
                        rs.getString("department"),
                        rs.getDouble("salary"),
                        rs.getString("city")
                );
            }

        } catch (SQLException e) {
            System.out.println("Error: " + e.getMessage());
        }
        return null;
    }

    @Override
    public void updateSalary(int empId, double newSalary) {
        String sql = "UPDATE employees SET salary=? WHERE emp_id=?";

        try (Connection con = DBConnection.getConnection();
             PreparedStatement ps = con.prepareStatement(sql)) {

            ps.setDouble(1, newSalary);
            ps.setInt(2, empId);

            int rows = ps.executeUpdate();

            if (rows > 0)
                System.out.println("Salary updated!");
            else
                System.out.println("Employee not found!");

        } catch (SQLException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }

    @Override
    public void deleteEmployee(int empId) {
        String sql = "DELETE FROM employees WHERE emp_id=?";

        try (Connection con = DBConnection.getConnection();
             PreparedStatement ps = con.prepareStatement(sql)) {

            ps.setInt(1, empId);

            int rows = ps.executeUpdate();

            if (rows > 0)
                System.out.println("Employee deleted!");
            else
                System.out.println("Employee not found!");

        } catch (SQLException e) {
            System.out.println("Error: " + e.getMessage());
        }
    }
}
