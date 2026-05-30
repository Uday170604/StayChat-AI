# RAG Evaluation Report

StayChat Hotel Q&A — retrieval + generation evaluation with metric workings.

**Mean Precision@7:** 0.381
**Mean Reciprocal Rank:** 1.0
**Retrieval:** hybrid dense (MiniLM + FAISS) + BM25, k=7 (k=10 for list-style queries)

## Q1: Which hotels have free WiFi and complimentary breakfast?

### Retrieved chunks
- **amen_004_chunk_0** (fused=0.817, dense=0.723, bm25=0.993, The Marina Grand): Free WiFi, complimentary breakfast for all room categories, beach cabanas, outdoor pool, kids club, and seafood grill.
- **amen_006_chunk_0** (fused=0.800, dense=0.692, bm25=1.000, Sunset Bay Beach Hotel): Free WiFi in rooms and beach area. Complimentary light breakfast (fruit, bread, tea). Beach loungers and water sports desk.
- **amen_003_chunk_0** (fused=0.708, dense=0.655, bm25=0.805, Kashi Riverside Resort): Complimentary WiFi, complimentary breakfast buffet, infinity pool, full-service spa, gym, private ghat access, and fine-dining restaurant.
- **amen_005_chunk_0** (fused=0.664, dense=0.643, bm25=0.703, City Centre Metro Hotel): Complimentary WiFi in lobby only; in-room WiFi is paid. Continental breakfast included with executive rooms only. Business centre and gym on site.
- **amen_007_chunk_0** (fused=0.694, dense=0.703, bm25=0.676, Hotel X Residency): High-speed WiFi complimentary in all rooms. Buffet breakfast included with standard booking. Fitness centre, laundry, and airport shuttle (paid).
- **amen_001_chunk_0** (fused=0.631, dense=0.624, bm25=0.644, Ganges View Heritage Hotel): Amenities include complimentary high-speed WiFi in all rooms and public areas. Guests receive a complimentary buffet breakfast daily from 6:30 to 10:00 AM in the ghat-view dining room.
- **amen_002_chunk_0** (fused=0.755, dense=0.750, bm25=0.766, Sacred Lotus Boutique Inn): Free WiFi throughout the property. Breakfast is available at the rooftop cafe for an additional charge (not complimentary). No spa; small fitness corner.

### Metric workings
- Precision@7 = hits / k = 5 / 7 → **0.7143**
- Relevant hits in top-7: `['amen_004_chunk_0', 'amen_006_chunk_0', 'amen_003_chunk_0', 'amen_007_chunk_0', 'amen_001_chunk_0']`
- Full relevant set: `['amen_001_chunk_0', 'amen_003_chunk_0', 'amen_004_chunk_0', 'amen_006_chunk_0', 'amen_007_chunk_0']`
- Reciprocal Rank = 1 / rank of first relevant chunk (0 if none) → **1.0** (first relevant at rank 1)

### LLM answer
*Generator: mock | hallucination controls: on | abstained: False*

Based on the retrieved documents:
- **The Marina Grand**: Free WiFi, complimentary breakfast for all room categories, beach cabanas, outdoor pool, kids club, and seafood grill. [amen_004_chunk_0]
- **Sunset Bay Beach Hotel**: Complimentary light breakfast (fruit, bread, tea). [amen_006_chunk_0]
- **Kashi Riverside Resort**: Complimentary WiFi, complimentary breakfast buffet, infinity pool, full-service spa, gym, private ghat access, and fine-dining restaurant. [amen_003_chunk_0]
- **Hotel X Residency**: High-speed WiFi complimentary in all rooms. [amen_007_chunk_0]
- **Ganges View Heritage Hotel**: Amenities include complimentary high-speed WiFi in all rooms and public areas. [amen_001_chunk_0]

### Qualitative analysis
**Retrieval**
- Retrieved 5/5 labeled-relevant chunks in top-7.
**Generation**
- Answer grounded with chunk citations; backend: **mock**.
- Ganges View Heritage included (WiFi + complimentary buffet breakfast).
**Improvement**
- Add cross-encoder reranker; optional metadata filter `amenity:wifi AND breakfast:complimentary`.

## Q2: What is the cancellation policy of Hotel X?

### Retrieved chunks
- **pol_001_chunk_0** (fused=0.830, dense=0.738, bm25=1.000, Hotel X Residency): Cancellation policy for Hotel X Residency: Free cancellation up to 48 hours before check-in. Cancellations within 48 hours incur one night charge. No-shows are charged 100% of the booking. Refunds processed within 7 busi...
- **desc_009_chunk_0** (fused=0.456, dense=0.451, bm25=0.463, Hotel X Residency): Hotel X Residency is a 4-star business hotel in central Varanasi near the railway station. Known for reliable service, meeting rooms, and express check-in for corporate guests.
- **pol_006_chunk_0** (fused=0.479, dense=0.599, bm25=0.256, City Centre Metro Hotel): Check-in 12:00 PM. Corporate bookings: 24-hour cancellation window. Leisure bookings: 48-hour free cancellation. Valid ID mandatory.
- **pol_007_chunk_0** (fused=0.383, dense=0.505, bm25=0.158, Sunset Bay Beach Hotel): Check-in 2:00 PM. Free cancellation up to 5 days before stay during monsoon season; otherwise 3 days prior. Refunds via UPI or card within 10 days.
- **pol_002_chunk_0** (fused=0.315, dense=0.404, bm25=0.150, Ganges View Heritage Hotel): Check-in 2:00 PM, check-out 11:00 AM. Cancellation: free until 72 hours prior; 50% charge within 72 hours. Pets not allowed. Government photo ID required at check-in.
- **pol_005_chunk_0** (fused=0.300, dense=0.373, bm25=0.164, Sacred Lotus Boutique Inn): Check-in 1:00 PM, check-out 10:00 AM. Cancellation within 24 hours: first night charged. Small pets under 10 kg allowed with prior approval.
- **pol_003_chunk_0** (fused=0.286, dense=0.356, bm25=0.156, The Marina Grand): Check-in 3:00 PM, check-out 12:00 PM. Flexible cancellation until 24 hours before arrival. Pet-friendly rooms available on request (surcharge applies). Passport or Aadhaar required.

### Metric workings
- Precision@7 = hits / k = 1 / 7 → **0.1429**
- Relevant hits in top-7: `['pol_001_chunk_0']`
- Full relevant set: `['pol_001_chunk_0']`
- Reciprocal Rank = 1 / rank of first relevant chunk (0 if none) → **1.0** (first relevant at rank 1)

### LLM answer
*Generator: mock | hallucination controls: on | abstained: False*

Cancellation policy for Hotel X Residency: Free cancellation up to 48 hours before check-in. Cancellations within 48 hours incur one night charge. No-shows are charged 100% of the booking. Refunds processed within 7 business days to the original payment method. [pol_001_chunk_0]

### Qualitative analysis
**Retrieval**
- Retrieved 1/1 labeled-relevant chunks in top-7.
- Top-ranked noise (e.g. `desc_009_chunk_0`): semantic overlap without satisfying all query constraints; hybrid BM25 + reranking penalizes paid-breakfast / low-rating false positives.
**Generation**
- Answer grounded with chunk citations; backend: **mock**.
- Hotel X policy answered from dedicated policy document (rank #1).
**Improvement**
- Entity linking on hotel name + policy category filter before generation.

## Q3: Suggest a hotel with excellent reviews near the beach.

### Retrieved chunks
- **rev_003_chunk_0** (fused=0.765, dense=0.639, bm25=1.000, The Marina Grand): Rating: 4.9/5. 'Excellent beach location steps from the sand. Kids loved the pool. Outstanding reviews for cleanliness and breakfast. Best beach hotel we have stayed in.' Anita & Rahul K., February 2025.
- **rev_009_chunk_0** (fused=0.683, dense=0.733, bm25=0.589, The Marina Grand): Rating: 3.8/5. 'Beach is crowded in peak season but hotel itself is excellent. Would still recommend for families.' Chris W., January 2025.
- **desc_005_chunk_0** (fused=0.597, dense=0.573, bm25=0.643, Sunset Bay Beach Hotel): Sunset Bay Beach Hotel is a 3-star property in South Goa with direct beach access. Known for sunset views and relaxed backpacker-luxe atmosphere.
- **rev_004_chunk_0** (fused=0.494, dense=0.488, bm25=0.505, Sunset Bay Beach Hotel): Rating: 4.2/5. 'Great value near the beach. Simple breakfast but fresh. Friendly staff.' Marco T., December 2024.
- **rev_005_chunk_0** (fused=0.265, dense=0.378, bm25=0.056, Kashi Riverside Resort): Rating: 4.7/5. 'Luxurious spa and flawless service. Worth the premium price.' Elena S., April 2025.
- **desc_004_chunk_0** (fused=0.443, dense=0.426, bm25=0.474, The Marina Grand): The Marina Grand is a 4-star beachfront resort in Goa, 50 metres from Calangute Beach. Family-friendly positioning with kids club and sea-view balconies. Ambiance: coastal, vibrant.
- **amen_006_chunk_0** (fused=0.428, dense=0.525, bm25=0.248, Sunset Bay Beach Hotel): Free WiFi in rooms and beach area. Complimentary light breakfast (fruit, bread, tea). Beach loungers and water sports desk.

### Metric workings
- Precision@7 = hits / k = 2 / 7 → **0.2857**
- Relevant hits in top-7: `['rev_003_chunk_0', 'desc_004_chunk_0']`
- Full relevant set: `['desc_004_chunk_0', 'loc_001_chunk_0', 'rev_003_chunk_0']`
- Reciprocal Rank = 1 / rank of first relevant chunk (0 if none) → **1.0** (first relevant at rank 1)

### LLM answer
*Generator: mock | hallucination controls: on | abstained: False*

I suggest **The Marina Grand**: 'Excellent beach location steps from the sand. Outstanding reviews for cleanliness and breakfast. [rev_003_chunk_0]

### Qualitative analysis
**Retrieval**
- Retrieved 2/3 labeled-relevant chunks in top-7.
- Top-ranked noise (e.g. `rev_009_chunk_0`): semantic overlap without satisfying all query constraints; hybrid BM25 + reranking penalizes paid-breakfast / low-rating false positives.
- Missed relevant chunks: `loc_001_chunk_0` — consider higher k, metadata filters, or cross-encoder reranking.
**Generation**
- Answer grounded with chunk citations; backend: **mock**.
- Recommendation prioritizes highest guest rating (4.9/5) over weaker 'excellent' wording at 3.8/5.
- Location chunk `loc_001` not in top-k; answer still valid from review + description context.
**Improvement**
- Boost `guest_review` + numeric rating features in reranker; fuse location chunks for beach proximity.

## Failure / edge case
**Query:** Which hotels allow pet tigers?
**Answer:** I don't have enough information in the provided documents to answer that.
**Retrieval**
- Partial recall: only 0/0 relevant chunks in top-7.
- Top-ranked noise (e.g. `pol_003_chunk_0`): semantic overlap without satisfying all query constraints; hybrid BM25 + reranking penalizes paid-breakfast / low-rating false positives.
**Generation**
- Faithful abstention when context cannot support the question.
**Improvement**
- Entity linking on hotel name + policy category filter before generation.
- Demonstrates hallucination control: query term *tigers* is not grounded in retrieved pet-policy text.
