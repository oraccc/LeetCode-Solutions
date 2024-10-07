const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('./database/leetcode.db');

// 初始化表结构
db.serialize(() => {
    db.run(`
        CREATE TABLE IF NOT EXISTS problems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            is_freq BOOLEAN DEFAULT 0
        )
    `);
});

// 插入或更新题目（如果数据库已有内容，覆盖）
function insertProblems(problems, callback) {
    const deleteQuery = `DELETE FROM problems`; // 先删除现有内容
    const insertQuery = `INSERT INTO problems (title, content, is_freq) VALUES (?, ?, ?)`;

    db.serialize(() => {
        db.run(deleteQuery, (err) => {
            if (err) {
                return callback(err);
            }

            const stmt = db.prepare(insertQuery);
            problems.forEach(problem => {
                stmt.run(problem.title, problem.content, problem.isFreq ? 1 : 0);
            });
            stmt.finalize(callback);
        });
    });
}

// 获取所有题目
function getAllProblems(callback) {
    const query = `SELECT * FROM problems`;
    db.all(query, [], (err, rows) => {
        if (err) {
            return callback(err);
        }
        callback(null, rows);
    });
}

module.exports = {
    insertProblems,
    getAllProblems
};