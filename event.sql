CREATE TABLE events (
    id INT AUTO_INCREMENT PRIMARY KEY,
    year INT NOT NULL,
    event_name VARCHAR(255),
    event_description TEXT,
    ia_type ENUM('High', 'Low'),
    format_type ENUM('Multimodal', 'Other'),
    link_to_resources TEXT
);
