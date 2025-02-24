const sqlite3 = require('sqlite3').verbose();
const db = new sqlite3.Database('./database/leetcode.db');

// 初始化表结构
db.serialize(() => {
    // db.run(`DROP TABLE IF EXISTS problems`);
    // db.run(`DROP TABLE IF EXISTS problem_attributes`);
    // 创建problems表，ID字段为字符串类型
    db.run(`
        CREATE TABLE IF NOT EXISTS problems (
            id TEXT PRIMARY KEY,
            content TEXT NOT NULL
        )
    `);
    // 创建problem_attributes表
    db.run(`
        CREATE TABLE IF NOT EXISTS problem_attributes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            problem_id TEXT,
            difficulty TEXT NOT NULL,
            FOREIGN KEY (problem_id) REFERENCES problems(id)
        )
    `);
});

// 插入题目（不会覆盖已有题目）
function insertProblems(problems, callback) {
    const insertQuery = `INSERT OR REPLACE INTO problems (id, content) VALUES (?, ?)`;

    db.serialize(() => {
        const stmt = db.prepare(insertQuery);
        problems.forEach(problem => {
            stmt.run(problem.id, problem.content);
        });
        stmt.finalize(callback);
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