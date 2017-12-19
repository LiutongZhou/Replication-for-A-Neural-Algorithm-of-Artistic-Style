var idx = lunr(function () {
  this.field('title', {boost: 10})
  this.field('excerpt')
  this.field('categories')
  this.field('tags')
  this.ref('id')
});



  
  
    idx.add({
      title: "Styletransfer",
      excerpt: "Styletransfer A Tensorflow implementation for A Neural Algorithm of Artistic Style (Gatys et al. 2015) Folder Structure ./ . ├──...",
      categories: ["None"],
      tags: [],
      id: 0
    });
    
  
    idx.add({
      title: "Home",
      excerpt: "Get notified when I add new stuff   @mmistakes Tip Me Super Customizable Everything from the menus, sidebars, comments, and...",
      categories: [],
      tags: [],
      id: 1
    });
    
  

  
  


console.log( jQuery.type(idx) );

var store = [
  
    
    
    
      
      {
        "title": "Styletransfer",
        "url": "http://localhost:4000/Replication-for-A-Neural-Algorithm-of-Artistic-Style/styletransfer/",
        "excerpt": "Styletransfer A Tensorflow implementation for A Neural Algorithm of Artistic Style (Gatys et al. 2015) Folder Structure ./ . ├──...",
        "teaser":
          
            null
          
      },
    
      
      {
        "title": "Home",
        "url": "http://localhost:4000/Replication-for-A-Neural-Algorithm-of-Artistic-Style/",
        "excerpt": "Get notified when I add new stuff   @mmistakes Tip Me Super Customizable Everything from the menus, sidebars, comments, and...",
        "teaser":
          
            null
          
      },
    
  
    
    
    
  ]

$(document).ready(function() {
  $('input#search').on('keyup', function () {
    var resultdiv = $('#results');
    var query = $(this).val();
    var result = idx.search(query);
    resultdiv.empty();
    resultdiv.prepend('<p class="results__found">'+result.length+' Result(s) found</p>');
    for (var item in result) {
      var ref = result[item].ref;
      if(store[ref].teaser){
        var searchitem =
          '<div class="list__item">'+
            '<article class="archive__item" itemscope itemtype="http://schema.org/CreativeWork">'+
              '<h2 class="archive__item-title" itemprop="headline">'+
                '<a href="'+store[ref].url+'" rel="permalink">'+store[ref].title+'</a>'+
              '</h2>'+
              '<div class="archive__item-teaser">'+
                '<img src="'+store[ref].teaser+'" alt="">'+
              '</div>'+
              '<p class="archive__item-excerpt" itemprop="description">'+store[ref].excerpt+'</p>'+
            '</article>'+
          '</div>';
      }
      else{
    	  var searchitem =
          '<div class="list__item">'+
            '<article class="archive__item" itemscope itemtype="http://schema.org/CreativeWork">'+
              '<h2 class="archive__item-title" itemprop="headline">'+
                '<a href="'+store[ref].url+'" rel="permalink">'+store[ref].title+'</a>'+
              '</h2>'+
              '<p class="archive__item-excerpt" itemprop="description">'+store[ref].excerpt+'</p>'+
            '</article>'+
          '</div>';
      }
      resultdiv.append(searchitem);
    }
  });
});
