package com.example.baselinechatbot

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import com.example.baselinechatbot.ui.theme.BaselineChatbotTheme


class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            BaselineChatbotTheme {
                Surface(color = MaterialTheme.colorScheme.background) {
                    ChatScreen()
                }
            }
        }
    }
}

