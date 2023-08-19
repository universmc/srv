<?php
include 'db_config.php';

// Établir la connexion
$conn = new mysqli($servername, $username, $password, $dbname);

if ($conn->connect_error) {
    die("Connection failed: " . $conn->connect_error);
}

$sql = "SELECT ID, Titre, `Meta-data`, Description, Prompt_Backend, Prompt_Frontend, Fichier_mapset, Code_Source FROM project_steps";
$result = $conn->query($sql);

$data = [];
if ($result->num_rows > 0) {
    while ($row = $result->fetch_assoc()) {
        $data[] = $row;
    }
    echo json_encode($data);
} else {
    echo json_encode([]);
}
$conn->close();
?>
