//Owns all state. sendMessage() appends user message, 
//calls the API in a coroutine, appends the response field, 
// or appends an error message on failure.

package com.example.baselinechatbot

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.launch

class ChatViewModel : ViewModel() {

    private val _messages = MutableStateFlow(
        listOf(
            ChatMessage(
                text = "Hello, I am your healthcare chatbot.",
                isUser = false,
                timestamp = getCurrentTime()
            )
        )
    )
    val messages: StateFlow<List<ChatMessage>> = _messages

    private val _isLoading = MutableStateFlow(false)
    val isLoading: StateFlow<Boolean> = _isLoading

    fun sendMessage(text: String) {
        if (text.isBlank()) return

        _messages.value = _messages.value + ChatMessage(
            text = text,
            isUser = true,
            timestamp = getCurrentTime()
        )
        _isLoading.value = true

        viewModelScope.launch {
            try {
                val result = RetrofitClient.api.query(QueryRequest(query = text))
                _messages.value = _messages.value + ChatMessage(
                    text = result.response,
                    isUser = false,
                    timestamp = getCurrentTime()
                )
            } catch (e: Exception) {
                _messages.value = _messages.value + ChatMessage(
                    text = "Could not reach the server. Please try again.",
                    isUser = false,
                    timestamp = getCurrentTime()
                )
            } finally {
                _isLoading.value = false
            }
        }
    }
}
