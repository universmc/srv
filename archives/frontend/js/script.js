document.addEventListener('DOMContentLoaded', function() {
    fetch('backend/api.php')
    .then(response => response.json())
    .then(data => {
        const tbody = document.getElementById('projectTable').getElementsByTagName('tbody')[0];
        data.forEach(row => {
            const tr = tbody.insertRow();
            for (const column in row) {
                const td = tr.insertCell();
                td.textContent = row[column];
            }
        });
    });
});
