
- The dataset is about crime cases in Boston area.
  -- each entry (row) is a crime event
  -- we care about locations, hence two columns---"Lat" and "Long"---are what we care.

- a bloom filter that captures dangerous locations.
  -- (this is a general description; not our learned version)
  -- inputs are a 2-element vector, [Lat, Long]
  -- output is a boolean: 0 -- not dangerous, and 1 -- dangerous
  -- if the input [lat0, long0] is in the dataset, the bloom filter must return 1
  -- if the input [lat1, long1] is not in the dataset, the bloom filter should return 0 most of the case (>99%)


- our "learned bloom filter"
  -- the inputs and outputs are the same as a general bloom filter (described above)
  -- the semantic of our learned boom filter is loosed:
    -- if the input [lat2, long2] is in the dataset, the learned bloom filter should return 1 in a probability of X%
    -- if the input [lat3, long3] is not in the dataset, the learned bloom filter should return 0 in a probability of Y%
    -- X% and Y% are defined by users

- Specs:
  [cheng: I borrow this from Changggeng's last version bloom filter; I would
  expect this "crime dataset" works much better]

  1. Start verification condition: 
      - Accuracy on all positive data should > 50% (namely, X%=50%)
  2. 80% inputs in Boston area (what is this area; see below) should return 0
     (namely, Y%=80%)

some notes:
  - we may want to start from tiny dataset, for example, 1000 entries
  - you can randomly generate "safe entries" by generating (lat, long) far away from dangerous places
  - boston area may be something like [42\~43, -72\~-71] (you can adjust this according to the data you've seen)
