const tableData = [
    ["1", "Formation HTML, CSS, JavaScript", "Utilisation de VS Code<br>Scripts: index.html, style.css, scripts.js", "Script, Fichier", "index.html, style.css, scripts.js"],
    ["2", "Soutenances et Evaluations", "Orales<br>Micro-projets: Sites web", "Présentation textuelle, Script", "--"],
    ["3", "Botnet GPT Codex", "Génération de scripts<br>Debugger<br>Normes W3C", "Script, Fichier", "Exemples à définir"],
    ["4", "Projets P1 à P12", "Échéance en avril 2024", "Présentation textuelle", "--"],
    ["5", "Calendrier de Travail", "Structure en ASCII<br>Planification mois/semaine/jour/heure", "Présentation textuelle", "Calendrier ASCII"],
    ["6", "Projet Allocation Universelle", "Cryptomonnaie<br>Smart contrats<br>Système de rémunération", "Script, Fichier, Présentation textuelle", "Exemples à définir"]
];

const tableBody = document.getElementById('tableBody');

tableData.forEach(rowData => {
    const row = document.createElement('tr');
    rowData.forEach(cellData => {
        const cell = document.createElement('td');
        cell.innerHTML = cellData;
        row.appendChild(cell);
    });
    tableBody.appendChild(row);
});
