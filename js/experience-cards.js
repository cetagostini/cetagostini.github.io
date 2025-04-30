// Experience cards popup functionality
document.addEventListener('DOMContentLoaded', function() {
  const toggleButtons = document.querySelectorAll('.experience-toggle');
  
  // Create overlay container for popups
  const overlay = document.createElement('div');
  overlay.className = 'experience-overlay';
  document.body.appendChild(overlay);
  
  // Create popup container
  const popup = document.createElement('div');
  popup.className = 'experience-popup';
  document.body.appendChild(popup);
  
  toggleButtons.forEach(button => {
    button.addEventListener('click', function() {
      const card = this.closest('.experience-card');
      const details = card.querySelector('.experience-details');
      const title = card.querySelector('.experience-title').cloneNode(true);
      const company = card.querySelector('.experience-company').cloneNode(true);
      const location = card.querySelector('.experience-location').cloneNode(true);
      const period = card.querySelector('.experience-period').cloneNode(true);
      const summary = card.querySelector('.experience-summary').cloneNode(true);
      const detailsContent = details.cloneNode(true);
      
      // Show overlay and popup
      overlay.classList.add('active');
      popup.classList.add('active');
      
      // Create close button
      const closeButton = document.createElement('button');
      closeButton.className = 'popup-close';
      closeButton.innerHTML = '×';
      
      // Clear previous content and add new content
      popup.innerHTML = '';
      popup.appendChild(closeButton);
      popup.appendChild(period);
      popup.appendChild(title);
      popup.appendChild(company);
      popup.appendChild(location);
      popup.appendChild(summary);
      popup.appendChild(detailsContent);
      
      // Show all details content in popup
      detailsContent.classList.add('expanded');
      
      // Add close functionality
      closeButton.addEventListener('click', function() {
        overlay.classList.remove('active');
        popup.classList.remove('active');
      });
      
      // Close popup when clicking outside
      overlay.addEventListener('click', function() {
        overlay.classList.remove('active');
        popup.classList.remove('active');
      });
    });
  });
}); 