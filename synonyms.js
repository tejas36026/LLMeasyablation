const nlp = require('compromise');
const axios = require('axios');

async function getMicrosoftRelatedNouns() {
    try {
        const word = 'Microsoft';
    
        // USE FULL HTML CONTENT INSTEAD OF EXTRACTS
        const wikiResponse = await axios.get('https://en.wikipedia.org/api/rest_v1/page/html/Microsoft');
        
        const wikiContent = wikiResponse.data;
        
        console.log('\n=== Wikipedia Content Length ===');
        console.log(`Total characters: ${wikiContent.length}`);

        // Extract ALL text from HTML (remove HTML tags)
        const textContent = wikiContent.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ');
        
        console.log(`Text-only characters: ${textContent.length}`);

        // Process with NLP
        const doc = nlp(textContent);

        // GET EVERYTHING - NO FILTERS
        const nouns = doc.nouns().out('array');
        const uniqueNouns = [...new Set(nouns)];
        
        // GET ALL ENTITIES WITHOUT FILTERS
        const people = doc.people().out('array');
        const uniquePeople = [...new Set(people)];
        
        const organizations = doc.organizations().out('array');
        const uniqueOrgs = [...new Set(organizations)];

        // Also get other entity types
        const places = doc.places().out('array');
        const uniquePlaces = [...new Set(places)];
        


        console.log('\n=== RAW COUNTS ===');
        console.log(`Total nouns: ${nouns.length}`);
        console.log(`Unique nouns: ${uniqueNouns.length}`);
        console.log(`People: ${people.length}`);
        console.log(`Organizations: ${organizations.length}`);
        console.log(`Places: ${places.length}`);

        

        // Return EVERYTHING
        return {
            originalWord: word,
            wikipediaContent: textContent.substring(0, 1000) + '...', // Sample
            wikipediaContentLength: textContent.length,
            wikipediaNouns: uniqueNouns,
            people: uniquePeople,
            organizations: uniqueOrgs,
            places: uniquePlaces,
       
            
            // Raw arrays (unfiltered)
            rawNouns: nouns,
            rawPeople: people,
            rawOrganizations: organizations,
            totalNounsFound: nouns.length,
            uniqueRelevantNouns: uniqueNouns.length
        };

    } catch (error) {
        console.error('Error:', error.message);
        return null;
    }
}

// Main execution
async function main() {
    console.log('🔍 Microsoft Related Nouns Finder - NO FILTERS\n');
    
    const result = await getMicrosoftRelatedNouns();
    
    if (result) {
        console.log('\n=== COMPLETE RESULTS ===');
        
        console.log('\n👥 ALL PEOPLE:');
        console.log(JSON.stringify(result.people, null, 2));
        
        console.log('\n🏢 ALL ORGANIZATIONS:');
        console.log(JSON.stringify(result.organizations, null, 2));
        
        console.log('\n📍 ALL PLACES:');
        console.log(JSON.stringify(result.places, null, 2));
        
    
        
        
        console.log(`\n📊 STATS:`);
        console.log(`Wikipedia content: ${result.wikipediaContentLength} characters`);
        console.log(`Total nouns: ${result.totalNounsFound}`);
        console.log(`Unique nouns: ${result.uniqueRelevantNouns}`);
        console.log(`People found: ${result.people.length}`);
        console.log(`Organizations found: ${result.organizations.length}`);
        console.log(`Places found: ${result.places.length}`);

        // Use:
        console.log(`\n📊 SUMMARY: ${result.wikipediaNouns.length} total nouns, ${result.people.length} people, ${result.organizations.length} organizations`);

        // AND actually show the nouns:
        console.log('\n=== ALL NOUNS ===');
        result.wikipediaNouns.forEach((noun, index) => {
            console.log(`${index + 1}. ${noun}`);
        });
        // Save complete data to file
        const fs = require('fs');
        fs.writeFileSync('complete_microsoft_data.json', JSON.stringify(result, null, 2));
        console.log('\n💾 Complete data saved to complete_microsoft_data.json');
    
        console.log('\n👥 ALL PEOPLE:');
        console.log(JSON.stringify(result.people, null, 2));
        
        console.log('\n🏢 ALL ORGANIZATIONS:');
        console.log(JSON.stringify(result.organizations, null, 2));
        
        console.log('\n📝 ALL NOUNS:');
        console.log(JSON.stringify(result.wikipediaNouns, null, 2));
        
        console.log(`\n📊 COMPLETE SUMMARY:`);
        console.log(`Total unique nouns: ${result.wikipediaNouns.length}`);
        console.log(`Total people: ${result.people.length}`);
        console.log(`Total organizations: ${result.organizations.length}`);
        console.log(`Wikipedia content length: ${result.wikipediaContentLength} chars`);

    
    }
}

// Run the script
if (require.main === module) {
    main();
}

module.exports = { getMicrosoftRelatedNouns };
