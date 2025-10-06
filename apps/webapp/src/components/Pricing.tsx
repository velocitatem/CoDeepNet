export default function Pricing() {
  const plans = [
    {
      name: "Open Source",
      price: "Free",
      features: [
        "All boilerplate code",
        "Docker configurations",
        "Basic ML setup",
        "Community support"
      ]
    },
    {
      name: "Pro",
      price: "$49/month",
      features: [
        "Everything in Open Source",
        "Advanced configurations",
        "Priority support",
        "Custom integrations"
      ]
    }
  ];

  return (
    <section>
      {/* TODO: Style this pricing section when implementing in your project */}
      <div>
        <h2>Pricing</h2>
        <div>
          {plans.map((plan, index) => (
            <div key={index}>
              <h3>{plan.name}</h3>
              <p>{plan.price}</p>
              <ul>
                {plan.features.map((feature, featureIndex) => (
                  <li key={featureIndex}>{feature}</li>
                ))}
              </ul>
              <button>Choose Plan</button>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}