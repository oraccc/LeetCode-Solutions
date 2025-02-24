const express = require('express');
const bodyParser = require('body-parser');
// const path = require('path');
const db = require('./db');
const app = express();

app.use(bodyParser.json({ limit: '10mb' }));
// 托管静态文件 (HTML, CSS, JS 等)
// app.use(express.static(path.join(__dirname)));
app.use(express.static("public"));

// API 路径：插入或覆盖题目
app.post('/api/update-problems', (req, res) => {
    console.log('Request received at /api/update-problems');
    // console.log('Request body:', req.body);

    const problems = req.body.problems;
    db.insertProblems(problems, (err) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.status(200).json({ message: 'Problems updated successfully' });
    });
});

// API 路径：获取所有题目
app.get('/api/problems', (req, res) => {
    db.getAllProblems((err, problems) => {
        if (err) {
            return res.status(500).json({ error: err.message });
        }
        res.status(200).json(problems);
    });
});

// 启动服务器
app.listen(3000, () => {
    console.log('Server is running on http://localhost:3000');
});