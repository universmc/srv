const seedrandom = require('seedrandom');
const fs = require('fs');

async function getIpFromProxy(
) {
  try {
    const readmeContent = fs.readFileSync('proxy.md', 'utf-8');
    const ipLine = readmeContent.split('\n').find(line => line.includes('ip='));

    if (!ipLine) {
      return null;
    }

    const ipMatch = ipLine.match(/ip="(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"/);
    if (ipMatch) {
      const ip = ipMatch[1];
      return ip;
    } else {
      console.error('Adresse IP introuvable dans proxy.md');
    }
  } catch (error) {
    console.error('Erreur lors de la lecture de proxy.md :', error);
  }
}

async function generateSeeds(
) {
  const ip = await getIpFromProxy();

  if (!ip) {
    console.error('Adresse IP non spécifiée dans proxy.md');
    return;
  }

  const seedrandomInstance = seedrandom(ip);

  const seedData = {
    seed1: seedrandomInstance.quick().toString(),
    seed2: seedrandomInstance.quick().toString(),
    seed3: seedrandomInstance.quick().toString(),
    seed4: seedrandomInstance.quick().toString(),
  };

  const seedCode = `module.exports = { ${Object.entries(seedData).map(([key, value]) => `${key}: '${value}'`).join(',\n')} };`;

  fs.writeFileSync('seeds.js', seedCode);

  // Écrire la nouvelle IP dans output/ip-seed.md
fs.writeFileSync('output/ip-seed.md', `ip="${ip}"`);
}

generateSeeds();
