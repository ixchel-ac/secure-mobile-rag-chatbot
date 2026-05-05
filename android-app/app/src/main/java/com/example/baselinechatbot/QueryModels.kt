//Models the full API contract. 
// top_k is hardcoded to 5. 
//Only response is shown in the UI but all fields are deserialized for completeness.

package com.example.baselinechatbot

data class QueryRequest(
    val query: String,
    val top_k: Int = 5
)

data class QueryResponse(
    val response: String,
    val redacted_entities: List<String>,
    val sources: List<String>,
    val fw_l2_passed: Boolean
)
