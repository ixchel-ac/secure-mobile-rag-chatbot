// Retrofit interface + singleton client. 
// HTTPS-only, TLS enforced by OkHttp's default SSLSocketFactory.

package com.example.baselinechatbot

import okhttp3.OkHttpClient
import okhttp3.logging.HttpLoggingInterceptor
import retrofit2.Retrofit
import retrofit2.converter.gson.GsonConverterFactory
import retrofit2.http.Body
import retrofit2.http.POST

interface ChatApiService {
    @POST("query")
    suspend fun query(@Body request: QueryRequest): QueryResponse
}

object RetrofitClient {
    // HTTPS enforces TLS 1.2+ on all Android API 24+ devices via OkHttp's default SSLSocketFactory.
    // No plain-text traffic is permitted — cleartext is not declared in the manifest.
    private const val BASE_URL =
        "https://mobile-rag-firewall-938910481811.us-west2.run.app/"

    private val loggingInterceptor = HttpLoggingInterceptor().apply {
        level = HttpLoggingInterceptor.Level.BASIC
    }

    private val client = OkHttpClient.Builder()
        .addInterceptor(loggingInterceptor)
        .build()

    val api: ChatApiService by lazy {
        Retrofit.Builder()
            .baseUrl(BASE_URL)
            .client(client)
            .addConverterFactory(GsonConverterFactory.create())
            .build()
            .create(ChatApiService::class.java)
    }
}
