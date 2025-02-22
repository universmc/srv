const natural = require('natural');

// Define intents and training phrases
const intents = {
  find_library: ['find a library for {{task}}', 'what library should i use for {{task}}'],
  get_documentation: ['how do i use {{library}}', 'where can i find the docs for {{library}}'],
  generate_code: ['generate code for {{task}}', 'i need code to {{task}}']
};

// Train the intent classifier
const classifier = new natural.BayesClassifier();
for (const intent in intents) {
  intents[intent].forEach(phrase => {
    classifier.addDocument(phrase, intent);
  });
}
classifier.train();

// Function to handle user input
function processInput(input) {
  const intent = classifier.classify(input);
  const entities = extractEntities(input, intents[intent]); // Extract entities from input

  // Call the appropriate action function
  switch (intent) {
    case 'find_library':
      return findLibrary(entities.task);
    case 'get_documentation':
      return getDocumentation(entities.library);
    case 'generate_code':
      return generateCode(entities.task);
    default:
      return "I'm not sure how to help with that.";
  }
}

// Action functions (replace with your actual logic)
function findLibrary(task) {
  // Search a database or list of libraries for the given task
  return `Try using electron-builder for ${task}`;
}

function getDocumentation(library) {
  // Fetch documentation for the given library
  return `Here's the documentation for ${library}: [link to docs]`;
}

function generateCode(task) {
  // Generate code for the given task
  return `Here's some code to get you started: \`console.log("Hello, world!");\``;
}

// Helper function to extract entities from user input
function extractEntities(input, phrase) {
  // Use NLP techniques (e.g., named entity recognition) to extract entities
  // For simplicity, we'll just use a basic string replacement for now
  const entities = {};
  const matches = phrase.match(/\{\{([^\}]+)\}\}/g);
  if (matches) {
    matches.forEach(match => {
      const entityName = match.replace(/\{\{([^\}]+)\}\}/, '$1');
      entities[entityName] = input.replace(phrase.replace(match, '(.*)'), '$1').trim();
    });
  }
  return entities;
}

module.exports = { processInput };