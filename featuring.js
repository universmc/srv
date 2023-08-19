fetch("/features/2023")
    .then(response => response.json())
    .then(data => {
        const timelineElement = document.getElementById("timeline");
        
        data.forEach(event => {
            const eventElement = document.createElement("div");
            eventElement.className = "event";
            eventElement.innerHTML = `
                <h2>${event.event_name} (${event.year})</h2>
                <p>${event.event_description}</p>
                <a href="${event.link_to_resources}">Learn more</a>
            `;
            timelineElement.appendChild(eventElement);
        });
    })
    .catch(error => {
        console.error("Error fetching data: ", error);
    });
