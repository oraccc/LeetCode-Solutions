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
        // 显示 id-title 格式
        link.textContent = problem.id;
        link.onclick = () => loadProblem(index, problems); // 点击加载对应题目内容
        listItem.appendChild(link);
        sidebar.appendChild(listItem);
    });
}

// 加载特定的题目到内容区域
function loadProblem(index, problems) {
    const titleBar = document.getElementById('title-bar');
    titleBar.textContent = problems[index].id; // 显示题目ID
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
        const lines = block.trim().split('\n'); // 将每个题目按行分割
        const id = lines[0].replace('##', '').trim(); // 获取第一行作为id，并去掉 `##`
        
        // 剩余部分作为content
        const content = lines.slice(1).join('\n').trim();
        
        return {
            id: id,
            content: marked.parse(content) // 使用 marked 解析剩余内容
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

