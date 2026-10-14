# Text Summarisation – Example Outputs

Showing the first **5** test-set examples from the real model evaluation run.

---

## Example 1

**Article snippet** (first 500 chars):

> (CNN) -- Scientists have discovered a new species of deep-sea fish near the Mariana Trench, the deepest part of the world's oceans. The creature, named Pseudoliparis swirei, was found at depths of more than 8,000 metres deeper than any fish recorded before. The discovery was made by a team from the University of Washington and Newcastle University during an expedition. Researchers believe the species can survive at these extreme depths because the high-pressure environment protects it from disea…

**Reference summary:**

> Scientists discover a new deep-sea fish species in the Mariana Trench at depths over 8,000 metres. The creature is named Pseudoliparis swirei and thrives in extreme pressure.

**TF-IDF:**

> The creature, named Pseudoliparis swirei, was found at depths of more than 8,000 metres deeper than any fish recorded before. The discovery was made by a team from the University of Washington and Newcastle University during an expedition. The findings were published in the journal Zootaxa, helping scientists understand how life adapts to extreme conditions.

**TextRank:**

> Scientists have discovered a new species of deep-sea fish near the Mariana Trench, the deepest part of the world's oceans. The creature, named Pseudoliparis swirei, was found at depths of more than 8,000 metres deeper than any fish recorded before. Researchers believe the species can survive at these extreme depths because the high-pressure environment protects it from diseases that affect fish at shallower depths.

**T5:**

> the species, named Pseudoliparis swirei, was found at depths of more than 8,000 metres deep than any fish recorded before. researchers believe the species can survive at these extreme depths because the high-pressure environment protects it from diseases that affect fish at shallower depths.

---

## Example 2

**Article snippet** (first 500 chars):

> (CNN) -- NASA's Perseverance rover has successfully collected its latest rock sample from Mars, providing scientists with crucial data about the planet's ancient geological history. The rover drilled into a boulder located on the rim of Jezero Crater, an area believed to have been submerged underwater billions of years ago. Mission specialists at the Jet Propulsion Laboratory confirmed that the core sample was safely sealed in a titanium tube. These samples are slated to be brought back to Earth…

**Reference summary:**

> NASA's Perseverance rover collects another Mars rock sample from Jezero Crater. The sample is sealed in a titanium tube for return to Earth in a future mission.

**TF-IDF:**

> The rover drilled into a boulder located on the rim of Jezero Crater, an area believed to have been submerged underwater billions of years ago. Mission specialists at the Jet Propulsion Laboratory confirmed that the core sample was safely sealed in a titanium tube. These samples are slated to be brought back to Earth by a future joint NASA-ESA Mars Sample Return campaign.

**TextRank:**

> NASA's Perseverance rover has successfully collected its latest rock sample from Mars, providing scientists with crucial data about the planet's ancient geological history. Mission specialists at the Jet Propulsion Laboratory confirmed that the core sample was safely sealed in a titanium tube. These samples are slated to be brought back to Earth by a future joint NASA-ESA Mars Sample Return campaign.

**T5:**

> the rover drilled into a boulder located on the rim of Jezero Crater. mission specialists at the Jet Propulsion Laboratory confirmed that the core sample was safely sealed in a titanium tube.

---

## Example 3

**Article snippet** (first 500 chars):

> (CNN) -- Renewable energy generation reached a historic milestone this year, accounting for more than thirty percent of global electricity production. The rapid growth was primarily driven by massive investments in solar photovoltaic arrays and offshore wind farms across Europe and Asia. According to the International Energy Agency report, solar power expanded at the fastest rate on record. Analysts highlight that declining equipment costs and supportive government policies are accelerating the…

**Reference summary:**

> Renewable energy now accounts for over thirty percent of global electricity generation. Solar and wind expansion is leading the global shift from fossil fuels.

**TF-IDF:**

> Renewable energy generation reached a historic milestone this year, accounting for more than thirty percent of global electricity production. According to the International Energy Agency report, solar power expanded at the fastest rate on record. Analysts highlight that declining equipment costs and supportive government policies are accelerating the transition away from fossil fuels.

**TextRank:**

> Renewable energy generation reached a historic milestone this year, accounting for more than thirty percent of global electricity production. The rapid growth was primarily driven by massive investments in solar photovoltaic arrays and offshore wind farms across Europe and Asia. According to the International Energy Agency report, solar power expanded at the fastest rate on record.

**T5:**

> the rapid growth was primarily driven by investments in solar photovoltaic arrays and offshore wind farms across Europe and Asia. analysts highlight declining equipment costs and supportive government policies are accelerating the transition away from fossil fuels.

---

## Example 4

**Article snippet** (first 500 chars):

> (CNN) -- A breakthrough in battery technology could soon double the driving range of electric vehicles while cutting charging times in half. Engineers at MIT developed a novel solid-state lithium-metal cell that replaces traditional liquid electrolytes with an ultra-thin ceramic separator. Testing shows the cell retains over ninety percent of its capacity after one thousand rapid charge cycles. Automotive manufacturers are already collaborating with the research team to scale production for comm…

**Reference summary:**

> MIT engineers develop a solid-state battery that could double EV driving range. The technology retains ninety percent capacity after 1,000 fast-charge cycles.

**TF-IDF:**

> A breakthrough in battery technology could soon double the driving range of electric vehicles while cutting charging times in half. Testing shows the cell retains over ninety percent of its capacity after one thousand rapid charge cycles. Automotive manufacturers are already collaborating with the research team to scale production for commercial adoption.

**TextRank:**

> A breakthrough in battery technology could soon double the driving range of electric vehicles while cutting charging times in half. Engineers at MIT developed a novel solid-state lithium-metal cell that replaces traditional liquid electrolytes with an ultra-thin ceramic separator. Testing shows the cell retains over ninety percent of its capacity after one thousand rapid charge cycles.

**T5:**

> lithium-metal cell replaces traditional liquid electrolytes with an ultra-thin ceramic separator. testing shows the cell retains over ninety percent of its capacity after one thousand rapid charge cycles.

---

## Example 5

**Article snippet** (first 500 chars):

> (CNN) -- Global biodiversity conservation efforts received a major boost with the establishment of a vast new marine protected area spanning over one hundred thousand square kilometers in the Pacific Ocean. The sanctuary will safeguard critical habitats for endangered sea turtles, hammerhead sharks, and migratory tuna populations. Commercial fishing, seabed mining, and oil exploration will be strictly prohibited within the reserve boundaries.…

**Reference summary:**

> A new 100,000 square kilometre marine sanctuary is established in the Pacific Ocean. Commercial fishing and mining are banned to protect endangered marine species.

**TF-IDF:**

> Global biodiversity conservation efforts received a major boost with the establishment of a vast new marine protected area spanning over one hundred thousand square kilometers in the Pacific Ocean. The sanctuary will safeguard critical habitats for endangered sea turtles, hammerhead sharks, and migratory tuna populations. Commercial fishing, seabed mining, and oil exploration will be strictly prohibited within the reserve boundaries.

**TextRank:**

> Global biodiversity conservation efforts received a major boost with the establishment of a vast new marine protected area spanning over one hundred thousand square kilometers in the Pacific Ocean. The sanctuary will safeguard critical habitats for endangered sea turtles, hammerhead sharks, and migratory tuna populations. Commercial fishing, seabed mining, and oil exploration will be strictly prohibited within the reserve boundaries.

**T5:**

> the sanctuary will safeguard critical habitats for endangered sea turtles, hammerhead sharks and migratory tuna populations. commercial fishing, seabed mining, and oil exploration will be strictly prohibited within the reserve boundaries.

---
