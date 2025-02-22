const fs = require("fs");
const Groq = require("groq-sdk");
const groq = new Groq();


async function main() {

  const regme = `
  **Intelligence artificielle au service de**

1 **Formation en intelligence artificielle : Nous allons élaborer une intelligence artificielle dédiée à aider les individus à atteindre leurs objectifs professionnels en s'appuyant sur leur Curriculum Vitae numérique universel (CVUN).
2 **Service de formation et de professionnalisation : Notre IA sera conçue pour assister les individus dans l'acquisition de nouvelles compétences et l'amélioration de leurs qualifications professionnelles.
3 **Monétisation des compétences : Nous créerons des mécanismes visant à valoriser et à monétiser les compétences des individus, en reconnaissant la valeur de leur travail et en leur offrant une rémunération adéquate.
4 **Intégration des besoins et des attentes des utilisateurs : Notre intelligence artificielle sera axée sur les besoins et les aspirations des utilisateurs, afin de les aider à améliorer leurs compétences et à atteindre leurs objectifs professionnels de manière ciblée et personnalisée.
5 **Monétisation de la valeur du travail : Nous mettrons en place un système de monétisation de la valeur du travail réalisé par les individus, en leur offrant une rémunération équitable pour les compétences qu'ils ont acquises et utilisées.
  `;
 const objectif = `Notre objectif est de former l'intelligence artificielle qui peut aider les individus à acquérir de nouvelles compétences, à valoriser la 
valeur du travail créé par leur travail et à monétiser leur curriculum vitae numérique universel

`;
  const completion = await groq.chat.completions.create({

    messages: [
      
      { role: "assistant", content: `${regme}+${objectif}`},
      {
        role: "system",
        content: `tu es Axιom.ia une intelligence artificielle de haut potentielle intégré au projet. Developpez le prompt Ultime: Présentation initiale ## votre {contexte}, ## votre {rôle},la régle ${regme} ## vos {compétences}, ## vos {tâches}, ## vos {fontions}, ## votre {routine}, ## les {processus}, ## les {caractéristiques}, ## ## les {Actions Immédiates} et ## le {resultat}{feedback} attentdu avec emoji's associé:`
      },

    ],
    model: "gemma2-9b-it",
    temperature: 0.5,
    max_tokens: 4096,
    }).then((chatCompletion)=>{
    const mdContent = chatCompletion.choices[0]?.message?.content;
    const outputFilePath = "Axiom" + new Date().toISOString().replace(/[-:TZ]/g, "") + ".md";
    fs.writeFileSync(outputFilePath, mdContent);
    console.log("Documentation du contructor généré et enregistré dans " + outputFilePath);
});
}

main();