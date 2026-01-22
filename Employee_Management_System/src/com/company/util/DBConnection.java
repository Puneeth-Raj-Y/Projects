package com.company.util;

import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.SQLException;

public class DBConnection {

    private static final String URL = 
        "jdbc:mysql://localhost:3306/company_db?useSSL=false&allowPublicKeyRetrieval=true";

    private static final String USER = "root";      // change if needed
    private static final String PASSWORD = "Mikey_1704";  // PUT YOUR MYSQL PASSWORD

    public static Connection getConnection() {
        try {
            return DriverManager.getConnection(URL, USER, PASSWORD);
        } catch (SQLException e) {
            throw new RuntimeException("Database connection failed!", e);
        }
    }
}
