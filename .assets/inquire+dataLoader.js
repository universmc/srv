const fs = require("fs");
const { prompt } = require("inquirer");

function startBrainstorming(
) {
    console.log("Début de la session de brainstorming...");

    const topic = getBrainstormingTopic();
    const ideas = generateIdeas(topic);
    filterAndPrioritizeIdeas(ideas);

    console.log("Fin de la session de brainstorming. Idées prioritaires :", ideas.filter(idea => idea.priority === "high"));
}

function getBrainstormingTopic(
) {
    const questions = [
        {
            type: "input",
            message: "Quel est le sujet principal de la session de brainstorming ?"
}
    ];

    return prompt(questions).topic;
}

function generateIdeas(topic) {
    const questions = [
        {
            type: "list",
            message: `Quelle idée pouvez-vous proposer pour ${topic} ?`,
            name: "idea"
}
    ];

    const ideaAnswers = prompt(questions);
    const ideas = [];

    while (ideaAnswers.idea !== "None") {
        ideas.push(ideaAnswers.idea);
        ideaAnswers = prompt(questions);
    }

    return ideas;
}

function filterAndPrioritizeIdeas(ideas) {
    console.log("Filtrage et priorité aux idées...");

    ideas.forEach(idea => {
        const questions = [
            {
                type: "list",
                message: `Quelle est la priorité pour l'idée "${idea}"?`,
                name: "priority",
                choices: ["High", "Medium", "Low"]
            }
        ];

        const { priority } = prompt(questions);
        console.log(`Idée "${idea}" priorisée comme ${priority}`);
    });
}

startBrainstorming();
