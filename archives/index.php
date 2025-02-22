<?php
include 'backend/db_config.php';

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
}
$conn->close();
?>

<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="frontend/styles.css">
    <title>Project Steps</title>
</head>
<body>
    <table id="projectTable">
        <thead>
            <tr>
                <th>ID</th>
                <th>Titre</th>
                <th>Meta-data</th>
                <th>Description</th>
                <th>Prompt_Backend</th>
                <th>Prompt_Frontend</th>
                <th>Fichier_mapset</th>
                <th>Code_Source</th>
            </tr>
        </thead>
        <tbody>
            <?php foreach ($data as $row): ?>
                <tr>
                    <td><?php echo $row['ID']; ?></td>
                    <td><?php echo $row['Titre']; ?></td>
                    <td><?php echo $row['Meta-data']; ?></td>
                    <td><?php echo $row['Description']; ?></td>
                    <td><?php echo $row['Prompt_Backend']; ?></td>
                    <td><?php echo $row['Prompt_Frontend']; ?></td>
                    <td><?php echo $row['Fichier_mapset']; ?></td>
                    <td><?php echo $row['Code_Source']; ?></td>
                </tr>
            <?php endforeach; ?>
        </tbody>
    </table>
    <script src="frontend/script.js"></script>
</body>
</html>
