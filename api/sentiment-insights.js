/**
 * POST /api/sentiment-insights — LLM screening of comments and summary generation.
 * Body: { posts: [ { id, title, body, barriers, comments: [{ id, body }] } ] }
 */
import OpenAI from 'openai';
import { getConfig, getApiKey } from '../lib/config.js';
import { resolveModel } from '../lib/chat-models.js';

export default async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(204).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ detail: 'Method not allowed' });
  }

  let body = req.body || {};

  const apiKey = getApiKey();
  if (!apiKey) {
    return res.status(500).json({ detail: 'API key not set. Set OPENAI_API_KEY in environment.' });
  }

    try {
      const config = getConfig();
      const llm = config.llm || {};
      const model = resolveModel(undefined, llm.model); 
      
      console.log(`Starting insights generation with model: ${model}, API key present: ${!!apiKey}`);

      const client = new OpenAI({
      apiKey,
      ...(llm.base_url ? { baseURL: llm.base_url } : {}),
    });

    const systemPrompt = `You are a Technology Strategy Analyst for Jaguar Land Rover (JLR) specializing in AEC (Architecture, Engineering, Construction) and manufacturing technologies.

You will be provided with a JSON array of Reddit posts and their top comments regarding a specific technology.

Your task is twofold:
1. "summary": Write a concise Markdown summary interpreting the core adoption barriers, challenges, AND potential opportunities/use-cases from these posts and comments. Focus strictly on relevance to enterprise adoption (like JLR).
2. "screened_comments": Review the comments for each post. Return the SINGLE most informative and insightful comment per post that provides real-world friction points, barriers, or practical opportunities/perspectives relevant to AEC/JLR. Exclude jokes, short agreements, or irrelevant noise. Max 1 comment per post. If no comment is genuinely insightful, return an empty array for that post.

Return ONLY valid JSON in this exact structure:
{
  "summary": "markdown string...",
  "screened_comments": {
    "post_id": [
      {
        "id": "comment_id",
        "body": "original comment text",
        "insight": "1-sentence explanation of why this barrier or opportunity is relevant to JLR"
      }
    ]
  }
}`;

    const userMessage = JSON.stringify(body.posts || [], null, 2);

          const resp = await client.chat.completions.create({
            model: model,
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: userMessage },
            ],
            temperature: 0.2,
            response_format: { type: "json_object" }
          });
          
          console.log("Sentiment Insights Response received");

          const responseText = (resp.choices[0]?.message?.content ?? '').trim();
          console.log("Raw LLM Response Length:", responseText.length);
          
          try {
            const parsed = JSON.parse(responseText);
            console.log("JSON parsed successfully");
            return res.status(200).json(parsed);
          } catch (parseError) {
            console.error("JSON Parsing Error:", parseError, "Response Text:", responseText.substring(0, 200) + "...");
            return res.status(500).json({ detail: "Failed to parse JSON response from LLM.", raw: responseText.substring(0, 1000) });
          }

        } catch (err) {
          console.error("Sentiment Insights Error:", err);
          const message = err.message || String(err);
          // Don't send status 500 here if headers are already sent, but let's just make sure we capture it
          return res.status(500).json({ detail: message, step: "Error" });
        }
}
