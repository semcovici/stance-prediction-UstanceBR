def list_to_sent(l):
  return ' '.join(l.replace('.', 'x')[1:-1].split(', '))