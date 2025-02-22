const fs = require('fs');

// Fonction pour compter les occurrences d'une lettre dans un fichier texte
function compterOccurrences(cheminFichier, lettre) {
    /**
     * Compte le nombre d'occurrences d'une lettre dans un fichier texte.
     *
     * Args:
     *     cheminFichier (string): Le chemin d'accès au fichier texte.
     *     lettre (string): La lettre à rechercher.
     *
     * Returns:
     *     int: Le nombre d'occurrences de la lettre.
     */
    const contenu = fs.readFileSync(cheminFichier, 'utf-8');
    return (contenu.split(lettre).length - 1);
}

// Exemple d'utilisation
const fichierTexte = 'output/test.md';

// Définir toutes les lettres et chiffres à rechercher
const caracteres = {
    "0": "0", "1": "1", "2": "2", "3": "3", "4": "4", "5": "5",
    "6": "6", "7": "7", "8": "8", "9": "9", "a": "a", "b": "b",
    "c": "c", "d": "d", "e": "e", "f": "f", "g": "g", "h": "h",
    "i": "i", "j": "j", "k": "k", "l": "l", "m": "m", "n": "n",
    "o": "o", "p": "p", "q": "q", "r": "r", "s": "s", "t": "t",
    "u": "u", "v": "v", "w": "w", "x": "x", "y": "y", "z": "z"
};

// Stocker les résultats des occurrences dans un objet
let resultats = {};

// Calculer le nombre d'occurrences pour chaque caractère
for (let [cle, caractere] of Object.entries(caracteres)) {
    resultats[cle] = compterOccurrences(fichierTexte, caractere);
}

// Enregistrer le dictionnaire dans un fichier JSON
fs.writeFileSync('data_phi.json', JSON.stringify(resultats, null, 2), 'utf-8');

console.log("Le fichier data_phi.json a été créé avec succès.");
