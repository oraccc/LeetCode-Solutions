// 加载并解析 Markdown 文件，提取题目标题和内容
function loadMdFile(filePath) {
    fetch(filePath)
        .then(response => response.text())
        .then(text => {
            // 调用解析函数，返回题目数组
            const problems = parseMarkdown(text);
            // 动态生成导航栏索引
            generateSidebar(problems);
            // 默认加载第一个题目
            loadProblem(0, problems);
        })
        .catch(error => {
            console.error('Error loading markdown file:', error);
        });
}


// 解析 Markdown 文件内容，返回题目数组
function parseMarkdown(mdText) {
    const problemBlocks = mdText.split('---'); // 按照 `---` 分割题目
    const problems = problemBlocks.map(block => {
        const titleMatch = block.match(/##\s*(.+)/); // 匹配 `##` 标题
        return {
            title: titleMatch ? titleMatch[1].trim() : '无标题',
            content: marked.parse(block.trim()) // 使用 marked 解析题目内容
        };
    });
    return problems;
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


// 点击按钮后，加载指定的 Markdown 文件
document.getElementById('load-file-btn').addEventListener('click', function() {
    const filePath = 'notes/freq_problems.md'; // 替换为你的 Markdown 文件路径
    loadMdFile(filePath);  // 加载并解析 Markdown 文件
});


// // 加载题目内容的函数
// function loadNote(noteId) {
//     // 显示加载状态
//     document.getElementById('note-content').innerHTML = 'Loading...';

//     // 动态加载对应的 markdown 文件
//     fetch(`notes/${noteId}.md`)
//         .then(response => response.text())
//         .then(text => {
//             // 使用 marked.js 将 markdown 转为 HTML
//             document.getElementById('note-content').innerHTML = marked.parse(text);
//             // 手动触发 highlight.js 的高亮
//             document.querySelectorAll('pre code').forEach((block) => {
//                 hljs.highlightBlock(block);
//             });
//         })
//         .catch(error => {
//             console.error('Error loading note:', error);
//             document.getElementById('note-content').innerHTML = 'Error loading content.';
//         });
// }