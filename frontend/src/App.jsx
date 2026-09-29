import { useState, useEffect } from 'react';
import { ShoppingBag, ShoppingCart, Trash2, ChevronDown, CheckCircle, Truck, FileText, ArrowRight, User, Plus, Minus, Tag, Target, Users, Flame, ChevronRight, Sparkles } from 'lucide-react';
import './index.css';

const API_BASE = 'http://localhost:8000';

const getCategoryPillClass = (category) => {
  if (!category) return 'pill-gray';
  const c = category.toLowerCase();
  if (c.includes('dairy')) return 'pill-blue';
  if (c.includes('snack') || c.includes('chocolate')) return 'pill-purple';
  if (c.includes('fresh') || c.includes('produce') || c.includes('fruit')) return 'pill-green';
  return 'pill-gray';
};

const getProductImage = (name) => {
  if (!name) return '📦';
  const n = name.toLowerCase();
  if (n.includes('milk')) return '🥛';
  if (n.includes('bun') || n.includes('bread')) return '🍞';
  if (n.includes('chocolate') || n.includes('dairy milk')) return '🍫';
  if (n.includes('dahi') || n.includes('yogurt')) return '🥣';
  if (n.includes('banana')) return '🍌';
  if (n.includes('apple')) return '🍎';
  if (n.includes('chips')) return '🥔';
  return '🛍️';
};

function App() {
  const [users, setUsers] = useState([]);
  const [selectedUser, setSelectedUser] = useState('');
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const [cartIds, setCartIds] = useState([]);
  const [cartDetails, setCartDetails] = useState(null);
  const [recommendations, setRecommendations] = useState([]);
  const [loading, setLoading] = useState(true);

  // Load users
  useEffect(() => {
    fetch(`${API_BASE}/api/users`)
      .then(res => res.json())
      .then(data => {
        setUsers(data.users);
        if (data.users.length > 0) {
          setSelectedUser(data.users[0]);
        }
      })
      .catch(err => console.error(err));
  }, []);

  // Load initial cart when user changes
  useEffect(() => {
    if (!selectedUser) return;
    setLoading(true);
    fetch(`${API_BASE}/api/user/${selectedUser}/initial-cart`)
      .then(res => res.json())
      .then(data => {
        setCartIds(data.cart_product_ids);
      })
      .catch(err => console.error(err));
  }, [selectedUser]);

  // Load cart details when cart changes
  useEffect(() => {
    if (!selectedUser) return;
    
    fetch(`${API_BASE}/api/cart/details`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: selectedUser,
        cart_product_ids: cartIds
      })
    })
      .then(res => res.json())
      .then(data => {
        setCartDetails(data);
        setLoading(false);
        
        if (!data.summary.threshold_unlocked) {
          fetchRecommendations();
        } else {
          setRecommendations([]);
        }
      })
      .catch(err => {
        console.error(err);
        setLoading(false);
      });
  }, [cartIds, selectedUser]);

  const fetchRecommendations = () => {
    fetch(`${API_BASE}/api/recommendations`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        user_id: selectedUser,
        cart_product_ids: cartIds
      })
    })
      .then(res => res.json())
      .then(data => setRecommendations(data.recommendations))
      .catch(err => console.error(err));
  };

  const handleClearCart = () => {
    setCartIds([]);
  };

  const handleIncrease = (productId) => {
    setCartIds([...cartIds, productId]);
  };

  const handleDecrease = (productId) => {
    const newCart = [...cartIds];
    const index = newCart.lastIndexOf(productId);
    if (index !== -1) {
      newCart.splice(index, 1);
      setCartIds(newCart);
    }
  };

  const handleDelete = (productId) => {
    setCartIds(cartIds.filter(id => id !== productId));
  };

  const getAggregatedCart = () => {
    if (!cartDetails || !cartDetails.items) return [];
    
    return cartDetails.items.reduce((acc, item) => {
      const existing = acc.find(i => i.product_id === item.product_id);
      if (existing) {
        existing.quantity += 1;
        existing.totalPrice += item.price;
      } else {
        acc.push({ ...item, quantity: 1, totalPrice: item.price });
      }
      return acc;
    }, []);
  };

  if (!users.length || !selectedUser) {
    return <div style={{display: 'flex', justifyContent: 'center', marginTop: '100px'}}><div className="loader" style={{borderColor: 'var(--border-color)', borderTopColor: 'var(--green-btn)'}}></div></div>;
  }

  const aggregatedCart = getAggregatedCart();
  const cartTotal = cartDetails?.summary.cart_total || 0;
  const threshold = cartDetails?.summary.free_delivery_threshold || 150;
  const gap = cartDetails?.summary.remaining_gap || 0;
  const unlocked = cartDetails?.summary.threshold_unlocked || false;
  const progressPercent = Math.min(100, (cartTotal / threshold) * 100);

  return (
    <div className="container">
      {/* Header */}
      <header className="app-header">
        <div className="header-left">
          <div className="header-icon">
            <ShoppingBag size={24} />
          </div>
          <div>
            <h1 className="header-title">Threshold Checkout</h1>
            <p className="header-subtitle">Smart recommendations to unlock free delivery</p>
          </div>
        </div>
        <div className="user-selector-wrapper">
          <div className="user-selector" onClick={() => setDropdownOpen(!dropdownOpen)}>
            <User size={16} />
            <span>{selectedUser}</span>
            <ChevronDown size={14} style={{ color: 'var(--text-main)', transform: dropdownOpen ? 'rotate(180deg)' : 'none', transition: 'transform 0.2s' }} />
          </div>
          {dropdownOpen && (
            <div className="custom-dropdown">
              {users.map(u => (
                <div 
                  key={u} 
                  className={`dropdown-item ${u === selectedUser ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedUser(u);
                    setDropdownOpen(false);
                  }}
                >
                  {u}
                </div>
              ))}
            </div>
          )}
        </div>
      </header>
      
      {/* Main Grid */}
      <div className="main-grid">
        {/* Left Column: Your Cart */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon">
                <ShoppingCart size={20} />
              </div>
              <div>
                <h2 className="card-title">Your Cart</h2>
                <p className="card-subtitle">{cartIds.length} item{cartIds.length !== 1 ? 's' : ''}</p>
              </div>
            </div>
            <button className="btn-outline" onClick={handleClearCart}>
              <Trash2 size={14} /> Clear Cart
            </button>
          </div>
          
          {loading && !cartDetails ? (
             <div style={{display: 'flex', justifyContent: 'center', padding: '2rem'}}><div className="loader" style={{borderColor: 'var(--border-color)', borderTopColor: 'var(--green-btn)'}}></div></div>
          ) : aggregatedCart.length === 0 ? (
            <div style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
              <ShoppingCart size={32} opacity={0.2} style={{ margin: '0 auto 1rem auto', display: 'block' }} />
              <p>Your cart is empty.</p>
            </div>
          ) : (
            <div className="cart-list">
              {aggregatedCart.map((item) => (
                <div key={item.product_id} className="cart-item">
                  <div className="item-img">
                    {getProductImage(item.product_name)}
                  </div>
                  <div className="item-info">
                    <div className="item-name">{item.product_name}</div>
                    <div className="item-cat">{item.category}</div>
                  </div>
                  
                  <div className="stepper">
                    <button className="stepper-btn" onClick={() => handleDecrease(item.product_id)}>
                      <Minus size={14} />
                    </button>
                    <span className="stepper-val">{item.quantity}</span>
                    <button className="stepper-btn" onClick={() => handleIncrease(item.product_id)}>
                      <Plus size={14} />
                    </button>
                  </div>
                  
                  <div className="item-price">₹{(item.price * item.quantity).toFixed(2)}</div>
                  
                  <button className="btn-delete" onClick={() => handleDelete(item.product_id)}>
                    <Trash2 size={16} />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Right Column: Order Summary */}
        <div className="card">
          <div className="card-header">
            <div className="card-title-group">
              <div className="card-icon" style={{color: 'var(--text-main)'}}>
                <FileText size={20} />
              </div>
              <div>
                <h2 className="card-title">Order Summary</h2>
              </div>
            </div>
          </div>
          
          {loading && !cartDetails ? (
            <div style={{display: 'flex', justifyContent: 'center', padding: '2rem'}}><div className="loader" style={{borderColor: 'var(--border-color)', borderTopColor: 'var(--green-btn)'}}></div></div>
          ) : cartDetails && (
            <>
              <div className="summary-row">
                <span>Cart Value</span>
                <span>₹{cartTotal.toFixed(2)}</span>
              </div>
              <div className="summary-row">
                <span>Free Delivery Threshold</span>
                <span>₹{threshold.toFixed(2)}</span>
              </div>
              {!unlocked && (
                <div className="summary-row gap">
                  <span>Remaining Gap</span>
                  <span>₹{gap.toFixed(2)}</span>
                </div>
              )}
              
              <div className="progress-wrapper">
                <div className="progress-bar-bg">
                  <div 
                    className={`progress-fill ${unlocked ? 'success' : ''}`}
                    style={{ width: `${progressPercent}%` }}
                  />
                </div>
                <div className="progress-labels">
                  <span>₹{cartTotal.toFixed(0)}</span>
                  <span>₹{threshold.toFixed(0)}</span>
                </div>
              </div>

              {unlocked ? (
                <div className="notification-box success">
                  <CheckCircle size={20} className="notification-icon" />
                  <div>
                    <div className="notification-title">Free Delivery Unlocked!</div>
                    <div className="notification-desc">You are eligible for free delivery on this order.</div>
                  </div>
                </div>
              ) : (
                <div className="notification-box">
                  <Truck size={20} className="notification-icon" />
                  <div>
                    <div className="notification-title">Add ₹{gap.toFixed(2)} more to unlock free delivery!</div>
                    <div className="notification-desc">Get free delivery on orders above ₹{threshold}.</div>
                  </div>
                </div>
              )}
              
              <button className="btn-primary">
                Proceed to Checkout <ArrowRight size={16} />
              </button>
            </>
          )}
        </div>
      </div>

      {/* Recommended for You */}
      {!loading && cartDetails && !unlocked && recommendations.length > 0 && (
        <div style={{ marginTop: '2rem' }}>
          <div className="rec-header">
            <div className="rec-title-group">
              <Sparkles size={20} className="rec-icon" />
              <div>
                <h2>Recommended for You</h2>
                <p className="rec-subtitle">Items that match your taste and help you reach the free delivery threshold</p>
              </div>
            </div>
            <button className="btn-link">
              View All <ChevronRight size={16} />
            </button>
          </div>
          
          <div className="rec-grid">
            {recommendations.map((rec, idx) => {
              // Deterministic reason icon based on index for the prototype
              let reasonIcon = <Target size={14} />;
              let reasonClass = "match";
              let reasonText = `Perfect match for your ₹${gap.toFixed(2)} gap`;
              
              if (idx === 1) {
                reasonIcon = <Users size={14} />;
                reasonClass = "popular";
                reasonText = "Popular with similar customers";
              } else if (idx === 2) {
                reasonIcon = <Tag size={14} />;
                reasonClass = "frequent";
                reasonText = "Frequently bought together";
              }

              return (
                <div key={rec.product_id} className="rec-card">
                  <div className="rec-top">
                    <div className="rec-img">
                      {getProductImage(rec.product_name)}
                    </div>
                    <div className="rec-info">
                      <span className={`pill ${getCategoryPillClass(rec.category)}`}>{rec.category}</span>
                      <h3 className="rec-name">{rec.product_name}</h3>
                      <div className="rec-price">₹{rec.price.toFixed(2)}</div>
                    </div>
                  </div>
                  
                  <div className="rec-reason">
                    <span className={`rec-reason-icon ${reasonClass}`}>{reasonIcon}</span>
                    {reasonText}
                  </div>
                  
                  <button className="btn-add" onClick={() => handleIncrease(rec.product_id)}>
                    <ShoppingCart size={14} /> Add to Cart
                  </button>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
