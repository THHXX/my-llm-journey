const fs = require('fs');
const path = require('path');
const https = require('https');

// 配置信息
const CONFIG = {
    ACCOUNT_ID: "13c3bae9344afd66af72fb968cba3a4f",
    API_TOKEN: "lz5mLetCUBCA5wBQJLIOJqfZiX0XvAhke19lYaxa",
    IMAGE_MODEL: "@cf/stabilityai/stable-diffusion-xl-base-1.0",
    TEXT_MODEL: "@cf/meta/llama-3-8b-instruct"
};

/**
 * 调用 LLM 翻译提示词
 */
async function translatePrompt(text) {
    // 简单检测是否包含中文
    if (!/[\u4e00-\u9fa5]/.test(text)) {
        return text;
    }

    console.log(`[JS] 检测到中文，正在优化提示词...`);
    const url = `https://api.cloudflare.com/client/v4/accounts/${CONFIG.ACCOUNT_ID}/ai/run/${CONFIG.TEXT_MODEL}`;

    const messages = [
        {
            role: "system",
            content: "You are an expert AI art prompt generator. Your task is to translate Chinese descriptions into high-quality, detailed English prompts for Stable Diffusion XL.\n\nRules:\n1. Translate the core meaning accurately.\n2. Enhance the prompt with visual details (lighting, artistic style, composition, texture) to make it suitable for high-quality image generation.\n3. Output ONLY the final English prompt string. Do not include explanations, introductions, or quotes."
        },
        {
            role: "user",
            content: text
        }
    ];

    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${CONFIG.API_TOKEN}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ messages: messages })
        });

        const result = await response.json();
        if (!result.success) {
            console.warn(`[JS] 翻译失败，将使用原始提示词: ${JSON.stringify(result.errors)}`);
            return text;
        }
        
        const translated = result.result.response.trim();
        console.log(`[JS] 翻译结果: "${translated}"`);
        return translated;

    } catch (error) {
        console.warn(`[JS] 翻译请求出错: ${error.message}`);
        return text;
    }
}

/**
 * 调用 Cloudflare AI API 生成图片
 * @param {string} rawPrompt 原始提示词
 * @param {string} outputPath 输出路径
 */
async function generateImage(rawPrompt, outputPath) {
    // 1. 尝试翻译/优化提示词
    const prompt = await translatePrompt(rawPrompt);
    
    console.log(`[JS] 正在生成图片: "${prompt}"...`);
    
    const url = `https://api.cloudflare.com/client/v4/accounts/${CONFIG.ACCOUNT_ID}/ai/run/${CONFIG.IMAGE_MODEL}`;
    
    try {
        const response = await fetch(url, {
            method: "POST",
            headers: {
                "Authorization": `Bearer ${CONFIG.API_TOKEN}`,
                "Content-Type": "application/json"
            },
            body: JSON.stringify({ prompt: prompt })
        });

        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`API请求失败: ${response.status} ${response.statusText} - ${errorText}`);
        }

        const buffer = await response.arrayBuffer();
        
        // 确保输出目录存在
        const dir = path.dirname(outputPath);
        if (!fs.existsSync(dir)) {
            fs.mkdirSync(dir, { recursive: true });
        }

        fs.writeFileSync(outputPath, Buffer.from(buffer));
        console.log(`[JS] 图片已保存至: ${outputPath}`);
        
    } catch (error) {
        console.error(`[JS] 错误: ${error.message}`);
        process.exit(1);
    }
}

// 命令行参数处理
const args = process.argv.slice(2);
const promptIndex = args.indexOf('--prompt');
const outputIndex = args.indexOf('--output');

if (promptIndex === -1 || outputIndex === -1) {
    console.error("用法: node generator.js --prompt \"<提示词>\" --output \"<输出路径>\"");
    process.exit(1);
}

const prompt = args[promptIndex + 1];
const output = args[outputIndex + 1];

generateImage(prompt, output);
