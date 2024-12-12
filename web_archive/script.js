// 页面加载时，检查数据库并加载内容
document.addEventListener('DOMContentLoaded', () => {
    loadFromDatabase();
});

// 从数据库加载题目
function loadFromDatabase() {
    fetch('/api/problems')
        .then(response => response.json())
        .then(problems => {
            if (problems.length > 0) {
                generateSidebar(problems);
                loadProblem(0, problems); // 默认加载第一个题目
            } else {
                document.getElementById('note-content').innerHTML = '数据库中没有题目';
            }
        })
        .catch(error => {
            console.error('Error loading problems:', error);
        });
}

// 动态生成导航栏
function generateSidebar(problems) {
    const sidebar = document.getElementById('index');
    sidebar.innerHTML = ''; // 清空现有的导航内容

    problems.forEach((problem, index) => {
        const listItem = document.createElement('li');
        const link = document.createElement('a');
        link.href = '#';
        link.textContent = problem.title;
        link.onclick = () => loadProblem(index, problems); // 点击加载对应题目内容
        listItem.appendChild(link);
        sidebar.appendChild(listItem);
    });
}

// 加载特定的题目到内容区域
function loadProblem(index, problems) {
    const noteContent = document.getElementById('note-content');
    noteContent.innerHTML = problems[index].content; // 显示题目内容

    // 清除其他链接的高亮样式
    const links = document.querySelectorAll('#index a');
    links.forEach(link => link.classList.remove('active'));

    // 给当前点击的链接添加高亮样式
    links[index].classList.add('active');

    // 手动触发 highlight.js 的高亮
    document.querySelectorAll('pre code').forEach((block) => {
        hljs.highlightBlock(block);
    });
}

// 解析并覆盖数据库中的内容
document.getElementById('load-file-btn').addEventListener('click', () => {
    // fetch('notes/problems_small.md')
    fetch('notes/problems.md')
        .then(response => response.text())
        .then(mdText => {
            const problems = parseMarkdown(mdText);
            updateDatabase(problems);
        })
        .catch(error => {
            console.error('Error loading markdown file:', error);
        });
});

// 解析 Markdown 文件内容，返回题目数组
function parseMarkdown(mdText) {
    const problemBlocks = mdText.split('---'); // 按照 `---` 分割题目
    return problemBlocks.map(block => {
        const titleMatch = block.match(/##\s*(.+)/); // 匹配 `###` 标题
        return {
            title: titleMatch ? titleMatch[1].trim() : '无标题',
            content: marked.parse(block.trim()), // 使用 marked 解析题目内容
            isFreq: false // 默认为非高频题，可以根据需要修改
        };
    });
}

// 发送更新请求，将问题插入或更新到数据库
function updateDatabase(problems) {
    fetch('/api/update-problems', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ problems }),
    })
    .then(response => response.json())
    .then(data => {
        console.log(data.message);
        showNotification('题目已更新'); // 成功后显示弹窗
        loadFromDatabase(); // 重新加载数据库中的内容
    })
    .catch(error => {
        console.error('Error updating database:', error);
    });
}


// 显示弹窗的方法
function showNotification(message) {
    const notification = document.getElementById('notification');
    notification.textContent = message; // 设置弹窗内容
    notification.classList.remove('hidden'); // 移除隐藏样式
    notification.classList.add('show'); // 添加显示样式

    // 设置2秒后自动淡出
    setTimeout(() => {
        notification.classList.remove('show');
        setTimeout(() => {
            notification.classList.add('hidden');
        }, 500); // 等待淡出动画结束后再隐藏
    }, 2000);
}

// // 加载并解析 Markdown 文件，提取题目标题和内容
// function loadMdFile(filePath) {
//     fetch(filePath)
//         .then(response => response.text())
//         .then(text => {
//             // 调用解析函数，返回题目数组
//             const problems = parseMarkdown(text);
//             // 动态生成导航栏索引
//             generateSidebar(problems);
//             // 默认加载第一个题目
//             loadProblem(0, problems);
//         })
//         .catch(error => {
//             console.error('Error loading markdown file:', error);
//         });
// }


// // 解析 Markdown 文件内容，返回题目数组
// function parseMarkdown(mdText) {
//     const problemBlocks = mdText.split('---'); // 按照 `---` 分割题目
//     const problems = problemBlocks.map(block => {
//         const titleMatch = block.match(/##\s*(.+)/); // 匹配 `##` 标题
//         return {
//             title: titleMatch ? titleMatch[1].trim() : '无标题',
//             content: marked.parse(block.trim()) // 使用 marked 解析题目内容
//         };
//     });
//     return problems;
// }


// // 动态生成导航栏
// function generateSidebar(problems) {
//     const sidebar = document.getElementById('index');
//     sidebar.innerHTML = ''; // 清空现有的导航内容

//     problems.forEach((problem, index) => {
//         const listItem = document.createElement('li');
//         const link = document.createElement('a');
//         link.href = '#';
//         link.textContent = problem.title;
//         link.onclick = () => loadProblem(index, problems); // 点击加载对应题目内容
//         listItem.appendChild(link);
//         sidebar.appendChild(listItem);
//     });
// }


// // 加载特定的题目到内容区域
// function loadProblem(index, problems) {
//     const noteContent = document.getElementById('note-content');
//     noteContent.innerHTML = problems[index].content; // 显示题目内容

//     // 清除其他链接的高亮样式
//     const links = document.querySelectorAll('#index a');
//     links.forEach(link => link.classList.remove('active'));

//     // 给当前点击的链接添加高亮样式
//     links[index].classList.add('active');

//     // 手动触发 highlight.js 的高亮
//     document.querySelectorAll('pre code').forEach((block) => {
//         hljs.highlightBlock(block);
//     });
// }


// // 点击按钮后，加载指定的 Markdown 文件
// document.getElementById('load-file-btn').addEventListener('click', function() {
//     const filePath = 'notes/freq_problems.md'; // 替换为你的 Markdown 文件路径
//     loadMdFile(filePath);  // 加载并解析 Markdown 文件
// });

