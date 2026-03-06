// 平滑滚动导航
document.querySelectorAll('nav a').forEach(anchor => {
    anchor.addEventListener('click', function(e) {
        e.preventDefault();
        const targetId = this.getAttribute('href');
        const targetSection = document.querySelector(targetId);
        if (targetSection) {
            targetSection.scrollIntoView({
                behavior: 'smooth',
                block: 'start'
            });
        }
    });
});

// 滚动显示动画
const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -50px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.style.opacity = '1';
            entry.target.style.transform = 'translateY(0)';
        }
    });
}, observerOptions);

document.querySelectorAll('.card, .stat-box, .schedule-item, .honor-item, .timeline-item').forEach(el => {
    el.style.opacity = '0';
    el.style.transform = 'translateY(20px)';
    el.style.transition = 'opacity 0.6s, transform 0.6s';
    observer.observe(el);
});

// 移动端菜单
const navUl = document.querySelector('nav ul');
if (window.innerWidth <= 768) {
    navUl.style.flexDirection = 'column';
    navUl.style.alignItems = 'center';
}

// 留言提交
function submitMessage() {
    const name = document.getElementById('msg-name').value.trim();
    const content = document.getElementById('msg-content').value.trim();
    
    if (!content) {
        alert('请输入留言内容');
        return;
    }
    
    const now = new Date();
    const time = now.getFullYear() + '-' + 
        String(now.getMonth()+1).padStart(2,'0') + '-' + 
        String(now.getDate()).padStart(2,'0');
    
    fetch('/api/message', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: name || '匿名', content: content, time: time})
    })
    .then(r => r.json())
    .then(r => {
        if (r.success) {
            alert('留言成功！');
            location.reload();
        } else {
            alert('留言失败，请重试');
        }
    });
}

// 球迷论坛提交
function submitDiscussion() {
    const name = document.getElementById('disc-name').value.trim();
    const content = document.getElementById('disc-content').value.trim();
    
    if (!content) {
        alert('请输入讨论内容');
        return;
    }
    
    const now = new Date();
    const time = now.getFullYear() + '-' + 
        String(now.getMonth()+1).padStart(2,'0') + '-' + 
        String(now.getDate()).padStart(2,'0');
    
    fetch('/api/discussion', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({name: name || '匿名', content: content, time: time})
    })
    .then(r => r.json())
    .then(r => {
        if (r.success) {
            location.reload();
        } else {
            alert('发布失败，请重试');
        }
    });
}

// 回复讨论
function handleReply(event, index) {
    if (event.key === 'Enter') {
        const content = event.target.value.trim();
        if (!content) return;
        
        const name = document.getElementById('disc-name')?.value.trim() || '匿名';
        const now = new Date();
        const time = now.getFullYear() + '-' + 
            String(now.getMonth()+1).padStart(2,'0') + '-' + 
            String(now.getDate()).padStart(2,'0');
        
        fetch('/api/discussion/reply', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({index: index, name: name, content: content, time: time})
        })
        .then(r => r.json())
        .then(r => {
            if (r.success) {
                location.reload();
            }
        });
    }
}
