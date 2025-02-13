package com.example.weather.service;

import org.json.JSONObject;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import com.example.weather.dto.WeatherResponse;

@Service
public class WeatherService {

    public final RestTemplate restTemplate;

    @Value("${weather.api.url}")
    private String baseUrl;

    @Value("${weather.api.key}")
    private String apiKey;

    public WeatherService(RestTemplate restTemplate) {
        this.restTemplate = restTemplate;
    }

    public String getRawWeather(String city) {
        String url = createUrl(city);

        return restTemplate.getForObject(url, String.class);
    }

    public WeatherResponse getWeather(String city) {
        String url = createUrl(city);

        try {
            String response = restTemplate.getForObject(url, String.class);
            return mapToWeatherResponse(response, city);
        } catch (Exception e) {
            return new WeatherResponse(city, "N/A", "N/A", "N/A");
        }
    }

    private WeatherResponse mapToWeatherResponse(String jsonResponse, String city) {

        JSONObject json = new JSONObject(jsonResponse);

        String temperature = String.valueOf(json
                .getJSONObject("current")
                .getDouble("temp_c"));

        String windSpeed = String.valueOf(json
                .getJSONObject("current")
                .getDouble("wind_mph"));

        String humidity = String.valueOf(json
                .getJSONObject("current")
                .getDouble("humidity"));

        return new WeatherResponse(city, temperature, windSpeed, humidity);
    }

    private String createUrl(String city) {
        String url = baseUrl
                .replace("{city}", city)
                .replace("{key}", apiKey);
        return url;
    }
}
