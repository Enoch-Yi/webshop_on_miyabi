from gym.envs.registration import register

from webshop_env.envs.web_agent_text_env import WebAgentTextEnv

register(
  id='WebAgentTextEnv-v0',
  entry_point='webshop_env.envs:WebAgentTextEnv',
)
