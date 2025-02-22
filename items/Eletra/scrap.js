const axios = require('axios');
const cheerio = require('cheerio');
const fs = require('fs');

// URL à analyser
const url = "https://archive.org";

// Fonction pour extraire le texte de la page HTML
async function extractTextFromUrl(url) {
    try {
        // Envoyer la requête HTTP GET
        const response = await axios.get(url);
        const htmlContent = response.data;

        // Charger le contenu HTML avec Cheerio
        const $ = cheerio.load(htmlContent);
        let text = "";

        // Extraire le texte de chaque élément
        $('*').each((index, element) => {
            if ($(element).text()) {
                text += $(element).text();
            }
        });

        return text;
    } catch (error) {
        console.error("Erreur lors de la récupération du contenu:", error);
        return "";
    }
}

// Fonction pour nettoyer le texte
function cleanText(text) {
    return text.toLowerCase().replace(/[.,;]/g, "");
}

// Fonction principale pour exécuter le script
async function main() {
    const text = await extractTextFromUrl(url);
    const cleanedText = cleanText(text);

    // Écrire le texte nettoyé dans un fichier
    fs.writeFileSync("output/test.md", cleanedText, 'utf-8');

    // Afficher le texte nettoyé
    console.log(cleanedText);
}

// Exécuter la fonction principale
main();
